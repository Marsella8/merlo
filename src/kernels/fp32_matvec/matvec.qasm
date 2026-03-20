.include "../share/vc4inc/vc4.qinc"

# ============================================================
# FP32 GEMV Kernel: y = A * x
#
# Matrix A in panel layout: 16 rows per panel, column-major.
# Each column = 16 FP32 values = 64 bytes (one per SIMD lane).
# No decode needed — TMU returns FP32 weights directly.
#
# Inner loop: 4 instructions per column (pipelined, dual-issue):
#   fadd r3, r3, r0; mov tmu0_s, ra6   # accum prev + submit col i
#   add ra6, ra6, rb11; ldtmu0          # advance + receive col i-1
#   mov r1, unif                        # x[i-1]
#   fmul r0, r4, r1                     # weight[i-1] * x[i-1]
#
# Uniform stream per QPU:
#   [0] B             - panels assigned to this QPU
#   [1] N             - number of columns
#   [2] mat_gpu       - GPU bus addr of first panel
#   [3] out_gpu       - GPU bus addr of output
#   [4] panel_stride  - bytes per panel (N * 64)
#   [5] x_gpu         - GPU bus addr of x data in uniform stream
#   [6..6+N-1]        - x[0]..x[N-1] (FP32)
#
# Register allocation:
#   ra0  = B (panel counter)
#   ra1  = N (columns, constant)
#   ra2  = panel base address
#   ra3  = output pointer
#   ra4  = panel_stride
#   ra5  = x_gpu (for uniform reset)
#   ra6  = TMU read address
#   ra7  = column loop counter
#   rb11 = 64 (column stride)
#   rb12 = elem_num * 4 (per-lane byte offset)
#   r0   = pending fmul result (pipelined one column behind)
#   r1   = x value from uniform
#   r3   = output accumulator
# ============================================================

# ---- Read header uniforms ----
mov ra0, unif           # B
mov ra1, unif           # N
mov ra2, unif           # mat_gpu
mov ra3, unif           # out_gpu
mov ra4, unif           # panel_stride
mov ra5, unif           # x_gpu

# ---- Precompute constants ----
mov r0, elem_num
shl rb12, r0, 2         # per-lane byte offset = elem_num * 4
ldi rb11, 64            # column stride (16 floats * 4 bytes)

# =============== Outer loop over panels ===============
:batch_loop

    # Reset uniform pointer to x data
    mov unif_addr, ra5
    nop
    nop
    nop

    # TMU base = panel_base + per-lane offset
    add ra6, ra2, rb12

    # Zero accumulator and pending product
    ldi r3, 0
    ldi r0, 0

    # Submit first TMU read (column 0)
    mov tmu0_s, ra6
    add ra6, ra6, rb11

    # Loop counter = N - 1 (pipelined: col 0 result processed in loop)
    mov r2, ra1
    nop
    sub ra7, r2, 1
    nop
    mov r2, ra7
    sub.setf -, r2, 0
    brr.allz -, :final_read
    nop
    nop
    nop

    # =============== Pipelined inner loop ===============
    # Each iteration: accumulate col i-1, submit col i, receive col i-1
    :col_loop
        fadd r3, r3, r0; mov tmu0_s, ra6   # accum prev + submit col i
        add ra6, ra6, rb11; ldtmu0          # advance + receive col i-1
        mov r1, unif                        # x[i-1]
        fmul r0, r4, r1                     # weight[i-1] * x[i-1]

        sub.setf ra7, ra7, 1
        brr.anynz -, :col_loop
        nop
        nop
        nop

    # ---- Process last column ----
    :final_read
    fadd r3, r3, r0         # accumulate pending product
    nop; ldtmu0             # receive last column
    mov r1, unif            # x[N-1]
    fmul r0, r4, r1         # weight * x
    fadd r3, r3, r0         # final accumulate

    # ---- Write output via VPM + DMA (mutex-protected) ----
    mov -, mutex_acq
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, r3
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
    add ra3, ra3, r0            # output_addr += 64 (16 floats)

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
