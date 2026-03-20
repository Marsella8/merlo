.include "../share/vc4inc/vc4.qinc"

# ============================================================
# GPU Scale-Matrix Kernel: out[i] = in[i] * scalar
#
# Each QPU independently scales its assigned chunk of an FP32
# matrix.  Uses batched VPM writes (4 rows per mutex acquire)
# and deep TMU pipelining (4 requests in flight) to maximize
# memory throughput.
#
# Uniform stream per QPU:
#   [0] N_batches   number of 4-group batches for this QPU
#   [1] in_gpu      GPU bus addr of this QPU's input start
#   [2] out_gpu     GPU bus addr of this QPU's output start
#   [3] scalar      FP32 scale value
# ============================================================

# ---- Read uniforms ----
mov ra0, unif           # N_batches
mov r1, unif            # in_gpu
mov ra3, unif           # out_gpu
mov ra5, unif           # scalar

# ---- Constants ----
# Per-lane TMU byte offset: elem_num * 4
mov r0, elem_num
shl rb13, r0, 2

ldi rb12, 64            # group stride = 16 floats * 4 bytes
ldi rb14, 256           # batch output stride = 4 groups * 64 bytes

# ---- Initial TMU address ----
add r0, r1, rb13        # r0 = in_gpu + per-lane offset

# Skip if nothing to do
sub.setf -, ra0, 0
brr.allz -, :done
nop
nop
nop

# ---- Pre-fill TMU FIFO with 4 reads ----
# Each dual-issue: submit current addr to TMU, advance addr by 64
add r0, r0, rb12; mov tmu0_s, r0
add r0, r0, rb12; mov tmu0_s, r0
add r0, r0, rb12; mov tmu0_s, r0
add r0, r0, rb12; mov tmu0_s, r0

# Consume first TMU result (group 0 -> r4, available next instruction)
nop; ldtmu0

# If only 1 batch, skip main loop
sub.setf -, ra0, 1
brr.allz -, :last_batch
nop
nop
nop

sub ra0, ra0, 1         # loop counter = N_batches - 1

# =============== Main pipelined loop ===============
# Invariant at entry: r4 holds current group's data,
# TMU receive FIFO has the next 3 results queued.
# Each group: dual-issue fmul + ldtmu0 (consume next) + add (advance addr),
#             then submit future TMU read.
:batch_loop
    # Group 0: submit TMU for future group, then multiply current + consume next
    mov tmu0_s, r0
    add r0, r0, rb12; fmul ra8, r4, ra5; ldtmu0

    # Group 1
    mov tmu0_s, r0
    add r0, r0, rb12; fmul ra9, r4, ra5; ldtmu0

    # Group 2
    mov tmu0_s, r0
    add r0, r0, rb12; fmul ra10, r4, ra5; ldtmu0

    # Group 3
    mov tmu0_s, r0
    add r0, r0, rb12; fmul ra11, r4, ra5; ldtmu0

    # ---- Write 4 rows via VPM + DMA (mutex-protected) ----
    mov -, mutex_acq
    mov vw_setup, vpm_setup(4, 1, h32(0))
    mov vpm, ra8
    mov vpm, ra9
    mov vpm, ra10
    mov vpm, ra11
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(4, 16, dma_h32(0, 0))
    mov vw_addr, ra3
    mov -, vw_wait

    add ra3, ra3, rb14; mov mutex_rel, 0

    sub.setf ra0, ra0, 1
    brr.anynz -, :batch_loop
    nop
    nop
    nop

# =============== Last batch (drain FIFO, no new submits) ===============
:last_batch
    fmul ra8, r4, ra5; ldtmu0
    fmul ra9, r4, ra5; ldtmu0
    fmul ra10, r4, ra5; ldtmu0
    fmul ra11, r4, ra5

    mov -, mutex_acq
    mov vw_setup, vpm_setup(4, 1, h32(0))
    mov vpm, ra8
    mov vpm, ra9
    mov vpm, ra10
    mov vpm, ra11
    mov -, vw_wait
    ldi vw_setup, vdw_setup_0(4, 16, dma_h32(0, 0))
    mov vw_addr, ra3
    mov -, vw_wait
    mov mutex_rel, 0

:done
nop; thrend
nop
nop
