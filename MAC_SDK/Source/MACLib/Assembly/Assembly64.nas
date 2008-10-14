
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


            mov r8, rdx
;            and edx, 0xfffffff0
	    and r8b, 0xf
AdaptAddLoop:
            movdqa xmm0, [rcx]
            movdqu xmm1, [rdx]
            paddw  xmm0, xmm1
            movdqa [rcx], xmm0
            movdqa xmm2, [rcx + 16]
            movdqu xmm3, [rdx + 16]
            paddw  xmm2, xmm3
            movdqa [rcx + 16], xmm2
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
            movdqa xmm0, [rcx]
            movdqu xmm1, [rdx]
            psubw  xmm0, xmm1
            movdqa [rcx], xmm0
            movdqa xmm2, [rcx + 16]
            movdqu xmm3, [rdx + 16]
            psubw  xmm2, xmm3
            movdqa [rcx + 16], xmm2
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
            pxor    xmm7, xmm7
            mov     r9d, r8d
            and     r9d, 1
            shr     r8d, 1

LoopDotEven:
            movdqu  xmm0, [rcx] ;pA
            pmaddwd xmm0, [rdx] ;pB
            paddd   xmm7, xmm0
            movdqu  xmm1, [rcx + 16]
            pmaddwd xmm1, [rdx + 16]
            paddd   xmm7, xmm1
            movdqu  xmm2, [rcx + 32]
            pmaddwd xmm2, [rdx + 32]
            paddd   xmm7, xmm2
            movdqu  xmm3, [rcx + 48]
            pmaddwd xmm3, [rdx + 48]
            add     rcx, byte 64
            add     rdx, byte 64
            paddd   xmm7, xmm3
            dec     r8d
            jnz short LoopDotEven

            cmp     r9d, byte 0
            je      DotFinal

            movdqu  xmm0, [rcx] ;pA
            pmaddwd xmm0, [rdx] ;pB
            paddd   xmm7, xmm0
            movdqu  xmm1, [rcx + 16]
            pmaddwd xmm1, [rdx + 16]
            paddd   xmm7, xmm1

DotFinal:
            movdqa  xmm6, xmm7
            psrldq  xmm6, 8
            paddd   xmm7, xmm6
            movdqa  xmm6, xmm7
            psrldq  xmm6, 4
            paddd   xmm7, xmm6
            movd    eax, xmm7
            emms
endproc


;
; BOOL GetMMXAvailable ( void );
;

proc        GetMMXAvailable
            mov     eax, 1
endproc
