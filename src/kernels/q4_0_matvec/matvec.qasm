.include "../share/vc4inc/vc4.qinc"

# ============================================================
# Q4_0 GEMV Kernel: y = A * x
#
# Matrix A stored in retiled Q4_0 panel format.
# Panel = 16 rows (one per SIMD lane). Block = 32 columns.
# Each 320-byte block: [16 FP32 scales | 4 x 16 nibble words]
#   Section 0 (bytes 0-63):   FP32 scale per lane
#   Section 1 (bytes 64-127): nibble word 0 (cols 0-7)
#   Section 2 (bytes 128-191): nibble word 1 (cols 8-15)
#   Section 3 (bytes 192-255): nibble word 2 (cols 16-23)
#   Section 4 (bytes 256-319): nibble word 3 (cols 24-31)
#
# Dequantization: weight_i = (nibble_i - 8) * scale
# Factored:  block_result = scale * (sum(q*x) - 8*sum(x))
# Precompute 8*sum(x) per block on ARM, pass via uniform.
#
# Uniform stream per QPU:
#   [0] B             - panels assigned to this QPU
#   [1] num_blocks    - N/32 (column-blocks per row)
#   [2] mat_gpu       - GPU bus addr of first panel
#   [3] out_gpu       - GPU bus addr of output
#   [4] panel_stride  - bytes per panel (num_blocks * 320)
#   [5] x_gpu         - GPU bus addr of x data in uniform stream
#   [6..] For each block: x[0..31], 8*sum(x[0..31])
#         (33 uniforms per block, num_blocks blocks total)
#
# Register allocation:
#   ra0  = B (panel counter, decremented)
#   ra1  = num_blocks (constant)
#   ra2  = panel base address (advanced per panel)
#   ra3  = output pointer (advanced per panel)
#   ra4  = panel_stride (constant)
#   ra5  = x_gpu (constant, for uniform reset)
#   ra6  = TMU read address (advanced during block)
#   ra7  = block loop counter (decremented)
#   ra8  = output accumulator (FP32, across all blocks)
#   rb11 = 64 (TMU stride between sections)
#   rb12 = elem_num * 4 (per-lane byte offset)
#   rb13 = FP32 scale for current block
#   r0   = temp (extracted nibble -> float -> product)
#   r1   = temp (x value from uniform)
#   r2   = temp (nibble word, shifted right each nibble)
#   r3   = temp (block partial accumulator for raw q*x)
# ============================================================

# ---- Read header uniforms ----
mov ra0, unif           # B
mov ra1, unif           # num_blocks
mov ra2, unif           # mat_gpu
mov ra3, unif           # out_gpu
mov ra4, unif           # panel_stride
mov ra5, unif           # x_gpu

# ---- Precompute constants ----
mov r0, elem_num
shl rb12, r0, 2         # per-lane byte offset = elem_num * 4
ldi rb11, 64            # TMU section stride

# =============== Outer loop over panels ===============
:batch_loop

    # Reset uniform pointer to x data
    mov unif_addr, ra5
    nop
    nop
    nop

    # TMU base = panel_base + per-lane offset
    add ra6, ra2, rb12

    # Zero output accumulator
    ldi ra8, 0

    # Check num_blocks > 0
    mov r0, ra1
    sub.setf -, r0, 0
    brr.allz -, :write_output
    nop
    nop
    mov ra7, ra1                # set block counter in delay slot

    # =============== Inner loop: one Q4_0 block (32 cols) ===============
    :block_loop

        # ---- Submit scale + word0 TMU reads ----
        # TMU submit sends OLD ra6 value, then advances ra6 by 64.
        # FIFO: submit scale, submit word0, wait, receive scale.
        add ra6, ra6, rb11; mov tmu0_s, ra6   # submit scale addr
        ldi r3, 0                              # zero block accum (+ RAW gap)
        add ra6, ra6, rb11; mov tmu0_s, ra6   # submit word0 addr
        nop                                    # RAW gap for ra6
        nop                                    # TMU latency fill
        nop
        nop
        nop; ldtmu0                            # receive scale -> r4
        mov rb13, r4                           # save FP32 scale

        # ---- Word 0: submit word1, then receive word0 ----
        add ra6, ra6, rb11; mov tmu0_s, ra6   # TMU write only
        nop; ldtmu0                            # TMU read only (separate instruction)
        mov r2, r4
        .rep nib, 8
            and r0, r2, 15; mov r1, unif
            itof r0, r0
            shr r2, r2, 4; fmul r0, r0, r1
            fadd r3, r3, r0
        .endr

        # ---- Word 1: submit word2, then receive word1 ----
        add ra6, ra6, rb11; mov tmu0_s, ra6   # TMU write only
        nop; ldtmu0                            # TMU read only
        mov r2, r4
        .rep nib, 8
            and r0, r2, 15; mov r1, unif
            itof r0, r0
            shr r2, r2, 4; fmul r0, r0, r1
            fadd r3, r3, r0
        .endr

        # ---- Word 2: submit word3, then receive word2 ----
        add ra6, ra6, rb11; mov tmu0_s, ra6   # TMU write only
        nop; ldtmu0                            # TMU read only
        mov r2, r4
        .rep nib, 8
            and r0, r2, 15; mov r1, unif
            itof r0, r0
            shr r2, r2, 4; fmul r0, r0, r1
            fadd r3, r3, r0
        .endr

        # ---- Word 3: receive word3 ----
        nop; ldtmu0
        mov r2, r4
        .rep nib, 8
            and r0, r2, 15; mov r1, unif
            itof r0, r0
            shr r2, r2, 4; fmul r0, r0, r1
            fadd r3, r3, r0
        .endr

        # ---- Block finalization ----
        # r3 = sum(q_i * x_i), need: scale * (q_dot - 8*sum_x)
        mov r1, unif               # precomputed 8 * sum(x) for this block
        fsub r3, r3, r1            # q_dot - 8*sum_x = sum((q-8)*x)
        fmul r3, r3, rb13         # multiply by scale
        fadd ra8, ra8, r3          # accumulate to output

        # Block loop control
        sub.setf ra7, ra7, 1
        brr.anynz -, :block_loop
        nop
        nop
        nop

    # ---- Write output via VPM + DMA (mutex-protected) ----
    :write_output
    mov -, mutex_acq
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra8
    mov -, vw_wait

    ldi vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    mov vw_addr, ra3
    mov -, vw_wait
    mov mutex_rel, 0

    # Advance panel and output pointers
    mov r0, ra4
    nop
    add ra2, ra2, r0            # panel_base += panel_stride
    ldi r0, 64
    add ra3, ra3, r0            # output_addr += 64 (16 fp32 values)

    # Panel loop control
    sub.setf ra0, ra0, 1
    brr.anynz -, :batch_loop
    nop
    nop
    nop

# ---- End program ----
nop; thrend
nop
nop
