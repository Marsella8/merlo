.include "../share/vc4inc/vc4.qinc"

# ============================================================
# GPU Gating Kernel: z[i] = x[i] * y[i]
#
# Used in LLM MLP blocks: after SiLU(x_up), multiply element-wise
# by the gate projection y_gate before the down-projection.
#   z = SiLU(x_up) * y_gate
#
# This kernel does just the element-wise multiply on two fp32
# vectors, reading both via TMU and writing the result via VPM DMA.
#
# Uniform stream (per QPU):
#   [0] N_groups    number of 16-element groups for this QPU
#   [1] x_gpu       GPU bus addr of first input (e.g. SiLU output)
#   [2] y_gpu       GPU bus addr of second input (e.g. gate proj)
#   [3] out_gpu     GPU bus addr of output
#
# Pipeline design:
#   Both x[i] and y[i] are read via TMU0 (submitted back-to-back,
#   consumed in FIFO order). The next group's reads are submitted
#   after the multiply, giving the TMU the entire VPM write
#   sequence (~8 instructions) to prefetch.
# ============================================================

# ---- Read header uniforms ----
mov ra0, unif           # N_groups
mov ra1, unif           # x_gpu
mov ra2, unif           # y_gpu
mov ra3, unif           # out_gpu

# ---- Precompute constants ----
mov r0, elem_num
shl rb13, r0, 2         # per-lane byte offset: elem_num * 4
ldi rb12, 64            # stride = 16 * 4 bytes

# ---- TMU address setup (per-lane) ----
add ra4, ra1, rb13      # x TMU addr = x_gpu + lane_offset
add ra5, ra2, rb13      # y TMU addr = y_gpu + lane_offset

# ---- Submit first pair of TMU reads (group 0) ----
mov tmu0_s, ra4         # submit x[0]
add ra4, ra4, rb12
mov tmu0_s, ra5         # submit y[0]
add ra5, ra5, rb12

# ---- Loop counter ----
mov r0, ra0
sub.setf r0, r0, 1     # r0 = N_groups - 1; Z if N_groups == 1
brr.allz -, :final_group
nop
nop
nop
mov ra7, r0

# =============== Main loop ===============
:group_loop
    # Consume x[i] from TMU FIFO (submitted first, arrives first)
    nop; ldtmu0
    mov r0, r4              # r0 = x[i]

    # Consume y[i] from TMU FIFO
    nop; ldtmu0
    #                         r4 = y[i]

    # z[i] = x[i] * y[i]
    fmul ra8, r0, r4

    # Submit reads for NEXT group (pipeline: TMU fetches during VPM write)
    mov tmu0_s, ra4         # submit x[i+1]
    add ra4, ra4, rb12
    mov tmu0_s, ra5         # submit y[i+1]
    add ra5, ra5, rb12

    # ---- Write 16 fp32 results via VPM + DMA ----
    mov -, mutex_acq
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra8
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    mov vw_addr, ra3
    mov -, vw_wait
    mov mutex_rel, 0

    add ra3, ra3, rb12     # advance output pointer

    sub.setf ra7, ra7, 1
    brr.anynz -, :group_loop
    nop
    nop
    nop

# =============== Final group ===============
:final_group
    nop; ldtmu0
    mov r0, r4

    nop; ldtmu0

    fmul ra8, r0, r4

    mov -, mutex_acq
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra8
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    mov vw_addr, ra3
    mov -, vw_wait
    mov mutex_rel, 0

# ---- End program ----
nop; thrend
nop
nop
