.include "../share/vc4inc/vc4.qinc"

# ============================================================
# GPU Softmax Kernel — single or multi-vector, multi-QPU
#
# Each QPU independently processes its assigned vectors.
# For attention: 8 QPUs each handle 1 head's scores.
# For LM head: 1 QPU handles the full logit vector.
#
# Three passes per vector (L2 cache reuse on passes 2-3):
#   Pass 1: Find max(x)
#   Pass 2: Compute sum(exp(x - max))
#   Pass 3: Normalize and write output
#
# Uniform stream per QPU:
#   [0] num_vectors   vectors assigned to this QPU (0 = skip)
#   [1] N             elements per vector (multiple of 16)
#   [2] in_gpu        bus addr of first input vector for this QPU
#   [3] out_gpu       bus addr of first output vector for this QPU
# ============================================================

# ---- Read uniforms ----
mov ra0, unif           # num_vectors
mov ra1, unif           # N
mov ra2, unif           # in_gpu (current)
mov ra3, unif           # out_gpu (current)

# ---- Constants ----
mov r0, elem_num
shl rb10, r0, 2         # per-lane TMU offset: elem_num * 4
ldi rb13, 64            # chunk stride: 16 * 4
ldi rb12, 0x3FB8AA3B    # log2(e) = 1.44269504
ldi rb14, 0xFF800000    # -infinity
ldi rb11, 0x80104000    # vdw_setup_0 base for dynamic DMA
ldi rb16, 16            # VPM batch constant

# vec_stride = N * 4
mov r0, ra1
nop
shl rb15, r0, 2

# num_chunks = N / 16
shr ra4, ra1, 4
nop                     # gap for ra4

# Skip if num_vectors == 0
sub.setf -, ra0, 0
brr.allz -, :done
nop
nop
nop

# =============== Vector loop ===============
:vector_loop

    # Save bases for this vector
    mov ra5, ra2            # saved input base
    mov ra6, ra3            # saved output base (used as write ptr in pass 3)

    # ===== PASS 1: Find max =====
    add ra7, ra2, rb10      # TMU addr
    mov r0, rb14            # running max = -inf

    mov tmu0_s, ra7
    add ra7, ra7, rb13

    mov r2, ra4
    sub r2, r2, 1
    sub.setf -, r2, 0
    brr.allz -, :p1_final
    mov ra8, r2             # delay: counter
    nop
    nop

    :p1_loop
        add ra7, ra7, rb13; mov tmu0_s, ra7
        nop; ldtmu0
        fmax r0, r0, r4
        sub.setf ra8, ra8, 1
        brr.anynz -, :p1_loop
        nop
        nop
        nop

    :p1_final
    nop; ldtmu0
    fmax r0, r0, r4

    # Horizontal max (all 16 lanes)
    # NOP required between writing r0 and reading it for rotation
    nop
    mov r1, r0 >> 1
    fmax r0, r0, r1
    nop
    mov r1, r0 >> 2
    fmax r0, r0, r1
    nop
    mov r1, r0 >> 4
    fmax r0, r0, r1
    nop
    mov r1, r0 >> 8
    fmax r0, r0, r1
    mov ra9, r0             # ra9 = global max

    # ===== PASS 2: Sum of exp(x - max) =====
    add ra7, ra5, rb10
    ldi r0, 0               # sum = 0

    mov tmu0_s, ra7
    add ra7, ra7, rb13

    mov r2, ra4
    sub r2, r2, 1
    sub.setf -, r2, 0
    brr.allz -, :p2_final
    mov ra8, r2
    nop
    nop

    :p2_loop
        add ra7, ra7, rb13; mov tmu0_s, ra7
        nop; ldtmu0
        fsub r1, r4, ra9
        fmul r1, r1, rb12
        mov exp, r1
        nop                     # SFU wait 1
        sub.setf ra8, ra8, 1    # SFU wait 2
        fadd r0, r0, r4
        brr.anynz -, :p2_loop
        nop
        nop
        nop

    :p2_final
    nop; ldtmu0
    fsub r1, r4, ra9
    fmul r1, r1, rb12
    mov exp, r1
    nop
    nop
    fadd r0, r0, r4

    # Horizontal sum
    nop
    mov r1, r0 >> 1
    fadd r0, r0, r1
    nop
    mov r1, r0 >> 2
    fadd r0, r0, r1
    nop
    mov r1, r0 >> 4
    fadd r0, r0, r1
    nop
    mov r1, r0 >> 8
    fadd r0, r0, r1

    # 1/sum
    mov recip, r0
    nop
    nop
    mov ra10, r4            # ra10 = inv_sum

    # ===== PASS 3: Normalize + write =====
    add ra7, ra5, rb10      # reset TMU addr

    mov r2, ra4
    nop
    mov ra11, r2            # remaining chunks
    nop                     # gap for ra11

    :p3_batch
        min r2, ra11, rb16
        mov ra12, r2        # batch_size

        mov -, mutex_acq
        mov vw_setup, vpm_setup(0, 1, h32(0))

        sub r2, r2, 1
        mov ra8, r2

        mov tmu0_s, ra7
        add ra7, ra7, rb13

        nop
        sub.setf -, ra8, 0
        brr.allz -, :p3_final
        nop
        nop
        nop

        :p3_inner
            add ra7, ra7, rb13; mov tmu0_s, ra7
            nop; ldtmu0
            fsub r1, r4, ra9
            fmul r1, r1, rb12
            mov exp, r1
            nop
            sub.setf ra8, ra8, 1
            fmul r1, r4, ra10
            mov vpm, r1
            brr.anynz -, :p3_inner
            nop
            nop
            nop

        :p3_final
        nop; ldtmu0
        fsub r1, r4, ra9
        fmul r1, r1, rb12
        mov exp, r1
        nop
        nop
        fmul r1, r4, ra10
        mov vpm, r1

        # DMA
        mov -, vw_wait
        mov r2, ra12
        nop
        shl r0, r2, 23
        add vw_setup, r0, rb11
        mov vw_addr, ra6
        mov -, vw_wait
        mov mutex_rel, 0

        # Advance output
        shl r0, r2, 6
        nop
        add ra6, ra6, r0

        # Remaining
        sub ra11, ra11, r2
        nop
        sub.setf -, ra11, 0
        brr.anynz -, :p3_batch
        nop
        nop
        nop

    # --- Next vector ---
    sub.setf ra0, ra0, 1
    brr.anynz -, :vector_loop
    add ra2, ra5, rb15      # delay: next input
    add ra3, ra6, 0         # delay: next output (ra6 already advanced)
    nop

:done
nop; thrend
nop
nop
