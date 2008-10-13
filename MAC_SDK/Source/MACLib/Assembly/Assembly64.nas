
%include "Tools64.inc"
                 
segment_code

;
; void  Adapt ( short* pM, const short* pAdapt, int nDirection, int nOrder )
;
;   r9d    nOrder
;   r8d    nDirection
;   rdx    pAdapt
;   rcx    pM
;   [esp+ 0]    Return Address

            align 16
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
proc        Adapt

            shr  r9d, 4

            cmp  r8d, byte 0       ; nDirection
            jle  short AdaptSub

AdaptAddLoop:
            movq  mm0, [rcx]
            paddw mm0, [rdx]
            movq  [rcx], mm0
            movq  mm1, [rcx + 8]
            paddw mm1, [rdx + 8]
            movq  [rcx + 8], mm1
            movq  mm2, [rcx + 16]
            paddw mm2, [rdx + 16]
            movq  [rcx + 16], mm2
            movq  mm3, [rcx + 24]
            paddw mm3, [rdx + 24]
            movq  [rcx + 24], mm3
            add   rcx, byte 32
            add   rdx, byte 32
            dec   r9d
            jnz   AdaptAddLoop

            emms
            ret

            align 16
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop

AdaptSub:   je    short AdaptDone

AdaptSubLoop:
            movq  mm0, [rcx]
            psubw mm0, [rdx]
            movq  [rcx], mm0
            movq  mm1, [rcx + 8]
            psubw mm1, [rdx + 8]
            movq  [rcx + 8], mm1
            movq  mm2, [rcx + 16]
            psubw mm2, [rdx + 16]
            movq  [rcx + 16], mm2
            movq  mm3, [rcx + 24]
            psubw mm3, [rdx + 24]
            movq  [rcx + 24], mm3
            add   rcx, byte 32
            add   rdx, byte 32
            dec   r9d
            jnz   AdaptSubLoop

            emms
AdaptDone:

endproc

;
; int  CalculateDotProduct ( const short* pA, const short* pB, int nOrder )
;
;   [esp+12]    nOrder
;   [esp+ 8]    pB
;   [esp+ 4]    pA
;   [esp+ 0]    Return Address

            align   16
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop

proc        CalculateDotProduct

            shr     r8d, 4
            pxor    mm7, mm7

loopDot:    movq    mm0, [rcx] ;pA
            pmaddwd mm0, [rdx] ;pB
            paddd   mm7, mm0
            movq    mm1, [rcx +  8]
            pmaddwd mm1, [rdx +  8]
            paddd   mm7, mm1
            movq    mm2, [rcx + 16]
            pmaddwd mm2, [rdx + 16]
            paddd   mm7, mm2
            movq    mm3, [rcx + 24]
            pmaddwd mm3, [rdx + 24]
            add     rcx, byte 32
            add     rdx, byte 32
            paddd   mm7, mm3
            dec     r8d
            jnz     loopDot

            movq    mm6, mm7
            psrlq   mm7, 32
            paddd   mm6, mm7
            movd    eax, mm6
            emms
endproc

;
; int  CalculateDotProduct ( const short* pA, const short* pB, int nOrder )
;
;   [esp+12]    nOrder
;   [esp+ 8]    pB
;   [esp+ 4]    pA
;   [esp+ 0]    Return Address

            align   16
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop
            nop

proc        CalculateDotProductXMM

            shr     r8d, 4
            pxor    xmm7, xmm7

loopDotXMM:    movdqu    xmm0, [rcx] ;pA
            pmaddwd xmm0, [rdx] ;pB
            paddd   xmm7, xmm0
            movdqu    xmm1, [rcx +  16]
            pmaddwd xmm1, [rdx +  16]
;            paddd   xmm7, xmm1
;            movq    xmm2, [rcx + 32]
;            pmaddwd xmm2, [rdx + 32]
;            paddd   xmm7, mm2
;            movq    xmm3, [rcx + 48]
;            pmaddwd xmm3, [rdx + 48]
            add     rcx, byte 32
            add     rdx, byte 32
;            paddd   xmm7, xmm3
            paddd   xmm7, xmm1
            dec     r8d
            jnz     loopDotXMM

            movq    xmm5, xmm7
            psrldq  xmm5, 16
            movq    xmm4, xmm5
            psrlq   xmm5, 32
            movq    xmm6, xmm7
            psrlq   xmm7, 32
            paddd   xmm6, xmm4
            paddd   xmm6, xmm5
            paddd   xmm6, xmm7
            movd    eax, xmm6
            emms
endproc


;
; BOOL GetMMXAvailable ( void );
;

proc        GetMMXAvailable
            push rax
            push rcx
            push rdx
            push rbx
            pushfq
            pop     rax
            mov     rcx, rax
            xor     rax, 0x200000
            push    rax
            popfq
            pushfq
            pop     rax
            cmp     rax, rcx
            jz      short return        ; no CPUID command, so no MMX

            mov     rax,1
            CPUID
            test    rdx,0x800000
return:     pop rbx
            pop rdx
            pop rcx
            pop rax
            setnz   al
            and     eax, byte 1
endproc

;            end
