.include "../share/vc4inc/vc4.qinc"

# ============================================================
# GPU SiLU Kernel: y[i] = x[i] * sigmoid(x[i])
#                       = x[i] / (1 + exp(-x[i]))
#
# Uses SFU hardware:
#   exp register  : computes 2^x  (NOT e^x)
#   recip register: computes 1/x
#
# So: exp(-x) = 2^(-x * log2(e))
#     sigmoid(x) = 1 / (1 + 2^(-x * log2(e)))
#     SiLU(x)    = x * sigmoid(x)
#
# Uniform stream (per QPU):
#   [0] N_groups    number of 16-element groups for this QPU
#   [1] in_gpu      GPU bus addr of this QPU's input start
#   [2] out_gpu     GPU bus addr of this QPU's output start
#
# Data layout:
#   Input/output are contiguous fp32 arrays in SDRAM.
#   Each TMU read fetches 16 fp32 values (64 bytes) — one per
#   SIMD lane — with per-lane addressing: base + elem_num * 4.
#   Stride between groups is 64 bytes.
#
# Pipeline design:
#   TMU reads are pipelined: the read for group i+1 is submitted
#   during group i's SFU EXP2 wait slots. The TMU result for the
#   next iteration is consumed in the branch delay slot (ldtmu0).
#   Mutex acquisition overlaps with SFU RECIP latency.
# ============================================================

# ---- Read header uniforms ----
mov ra0, unif           # N_groups
mov ra1, unif           # in_gpu
mov ra2, unif           # out_gpu

# ---- Precompute constants ----

# Per-lane TMU byte offset: elem_num * 4
# Lane k reads from base_addr + k*4, giving 16 contiguous fp32 values
mov r0, elem_num
shl rb13, r0, 2         # rb13 = lane_offset

# Float constants (IEEE 754 bit patterns)
ldi rb10, 0xBFB8AA3B    # -log2(e) = -1.4426950... (multiply x by this to get -x*log2(e))
ldi rb11, 0x3F800000    # 1.0

# ---- TMU address setup ----
add ra3, ra1, rb13      # TMU addr = in_gpu + per-lane offset
ldi rb12, 64            # stride (placed here to fill ra3 write-read hazard gap)

# ---- Submit first TMU read (group 0) and consume it ----
mov tmu0_s, ra3
add ra3, ra3, rb12      # advance to group 1
nop; ldtmu0             # consume group 0 result -> r4 (available next instr)

# ---- Check if only 1 group (skip loop) ----
mov r0, ra0
sub.setf r0, r0, 1     # r0 = N_groups - 1; Z flag set if N_groups == 1
brr.allz -, :final_group
nop
nop
nop
mov ra7, r0             # loop counter = N_groups - 1

# =============== Main loop (groups 0 through N_groups-2) ===============
# Invariant at loop entry: r4 holds the TMU result for the current group
:group_loop
    # r4 = TMU data for current group
    mov r0, r4              # r0 = x (preserve for final multiply)

    # Step 1: compute -x * log2(e)
    fmul r1, r0, rb10

    # Step 2: SFU EXP2 → 2^(-x*log2(e)) = exp(-x)
    mov exp, r1

    # Fill 2 SFU wait slots with TMU management for NEXT group:
    mov tmu0_s, ra3         # [EXP wait 1]: submit TMU read for next group
    add ra3, ra3, rb12      # [EXP wait 2]: advance TMU pointer

    # r4 = exp(-x) (SFU result now available)

    # Step 3: 1 + exp(-x)
    fadd r1, r4, rb11

    # Step 4: SFU RECIP → 1/(1 + exp(-x)) = sigmoid(x)
    mov recip, r1

    # Fill SFU wait slots: loop counter + mutex acquire
    sub.setf ra7, ra7, 1   # [RECIP wait 1]: decrement loop counter, set flags
    mov -, mutex_acq        # [RECIP wait 2]: acquire mutex (stall overlaps SFU latency)

    # r4 = sigmoid(x) (SFU result now available)

    # Step 5: x * sigmoid(x) = SiLU(x)
    fmul ra8, r0, r4

    # ---- Write 16 fp32 results via VPM + DMA (already hold mutex) ----
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra8
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    mov vw_addr, ra2
    mov -, vw_wait
    mov mutex_rel, 0

    # Branch back if more groups remain
    brr.anynz -, :group_loop
    add ra2, ra2, rb12     # [delay 1]: advance output pointer (always executes)
    nop                    # [delay 2]
    nop; ldtmu0            # [delay 3]: consume next TMU result -> r4

# Fell through: last iteration's delay slots executed.
# - ra2 advanced to correct position for final group
# - ldtmu0 consumed the last submitted TMU result into r4

# =============== Final group (group N_groups-1) ===============
:final_group
    mov r0, r4

    fmul r1, r0, rb10
    mov exp, r1
    nop
    nop

    fadd r1, r4, rb11
    mov recip, r1
    nop
    nop

    fmul ra8, r0, r4

    mov -, mutex_acq
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra8
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    mov vw_addr, ra2
    mov -, vw_wait
    mov mutex_rel, 0

# ---- End program ----
nop; thrend
nop
nop
