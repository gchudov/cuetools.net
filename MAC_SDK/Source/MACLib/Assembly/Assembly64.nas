
%include "Tools64.inc"
                 
segment_code

%imacro AdaptAddAligned 1
AdaptAddLoop0%1:
            movdqa xmm0, [rcx]
            movdqa xmm1, [rdx]
            %1     xmm0, xmm1
            movdqa [rcx], xmm0
            movdqa xmm2, [rcx + 16]
            movdqa xmm3, [rdx + 16]
            %1     xmm2, xmm3
            movdqa [rcx + 16], xmm2
            add   rcx, byte 32
            add   rdx, byte 32
            dec   r9d
            jnz   AdaptAddLoop0%1
            emms
            ret
%endmacro

%imacro AdaptAddUnaligned 1
            mov     r8d, r9d
            and     r8d, 1
            shr     r9d, 1
            cmp     r9d, byte 0
            je      short AdaptAddLoopULast%1
AdaptAddLoopU%1:
            movdqa xmm0, [rcx]
            movdqu xmm1, [rdx]
            %1     xmm0, xmm1
            movdqa [rcx], xmm0
            movdqa xmm2, [rcx + 16]
            movdqu xmm3, [rdx + 16]
            %1     xmm2, xmm3
            movdqa [rcx + 16], xmm2
            movdqa xmm4, [rcx+32]
            movdqu xmm5, [rdx+32]
            %1     xmm4, xmm5
            movdqa [rcx+32], xmm4
            movdqa xmm6, [rcx + 48]
            movdqu xmm7, [rdx + 48]
            %1     xmm6, xmm7
            movdqa [rcx + 48], xmm6
            add   rcx, byte 64
            add   rdx, byte 64
            dec   r9d
            jnz   AdaptAddLoopU%1
AdaptAddLoopULast%1:
            cmp   r8d, byte 0
            je    short AdaptAddLoopUEnd%1
            movdqa xmm0, [rcx]
            movdqu xmm1, [rdx]
            %1     xmm0, xmm1
            movdqa [rcx], xmm0
            movdqa xmm2, [rcx + 16]
            movdqu xmm3, [rdx + 16]
            %1     xmm2, xmm3
            movdqa [rcx + 16], xmm2
AdaptAddLoopUEnd%1:
            emms
            ret
%endmacro

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
            jle AdaptSub1

AdaptAddUnaligned paddw

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

AdaptSub1:   je AdaptDone1
AdaptAddUnaligned psubw
AdaptDone1:
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

            cmp     r8d, byte 0
            je      DotNonEven

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

DotNonEven:
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
