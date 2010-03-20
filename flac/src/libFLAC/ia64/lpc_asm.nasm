;  vim:filetype=nasm ts=8

;  libFLAC - Free Lossless Audio Codec library
;  Copyright (C) 2001,2002,2003,2004,2005,2006,2007,2008  Josh Coalson
;
;  Redistribution and use in source and binary forms, with or without
;  modification, are permitted provided that the following conditions
;  are met:
;
;  - Redistributions of source code must retain the above copyright
;  notice, this list of conditions and the following disclaimer.
;
;  - Redistributions in binary form must reproduce the above copyright
;  notice, this list of conditions and the following disclaimer in the
;  documentation and/or other materials provided with the distribution.
;
;  - Neither the name of the Xiph.org Foundation nor the names of its
;  contributors may be used to endorse or promote products derived from
;  this software without specific prior written permission.
;
;  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
;  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
;  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
;  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
;  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

;/*
; * MMX optimized FLAC DSP utils
; * Copyright (c) 2007 Loren Merritt
; *
; * This file is part of FFmpeg.
; *
; * FFmpeg is free software; you can redistribute it and/or
; * modify it under the terms of the GNU Lesser General Public
; * License as published by the Free Software Foundation; either
; * version 2.1 of the License, or (at your option) any later version.
; *
; * FFmpeg is distributed in the hope that it will be useful,
; * but WITHOUT ANY WARRANTY; without even the implied warranty of
; * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
; * Lesser General Public License for more details.
; *
; * You should have received a copy of the GNU Lesser General Public
; * License along with FFmpeg; if not, write to the Free Software
; * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
; */

%include "nasm.h"

	data_section

;cglobal FLAC__lpc_compute_autocorrelation_asm_ia64
cglobal FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_4
cglobal FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_8
cglobal FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_12

	code_section

; **********************************************************************
;
; void FLAC__lpc_compute_autocorrelation_asm(const FLAC__real data[], unsigned data_len, unsigned lag, FLAC__real autoc[])
; {
;	FLAC__real d;
;	unsigned sample, coeff;
;	const unsigned limit = data_len - lag;
;
;	FLAC__ASSERT(lag > 0);
;	FLAC__ASSERT(lag <= data_len);
;
;	for(coeff = 0; coeff < lag; coeff++)
;		autoc[coeff] = 0.0;
;	for(sample = 0; sample <= limit; sample++) {
;		d = data[sample];
;		for(coeff = 0; coeff < lag; coeff++)
;			autoc[coeff] += d * data[sample+coeff];
;	}
;	for(; sample < data_len; sample++) {
;		d = data[sample];
;		for(coeff = 0; coeff < data_len - sample; coeff++)
;			autoc[coeff] += d * data[sample+coeff];
;	}
; }
;

	ALIGN 16
cident FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_4
	;   r9d    == autoc[]
	;   r8d    == lag
	;   rdx    == data_len
	;   rcx    == data[]

	;ASSERT(lag > 0)
	;ASSERT(lag <= 4)
	;ASSERT(lag <= data_len)

	;	for(coeff = 0; coeff < lag; coeff++)
	;		autoc[coeff] = 0.0;
	xorps	xmm5, xmm5

	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[0]
	add	rcx, 4
	movaps	xmm2, xmm0			; xmm2 = 0,0,0,data[0]
	shufps	xmm0, xmm0, 0			; xmm0 == data[sample],data[sample],data[sample],data[sample] = data[0],data[0],data[0],data[0]
.warmup:					; xmm2 == data[sample-3],data[sample-2],data[sample-1],data[sample]
	mulps	xmm0, xmm2			; xmm0 = xmm0 * xmm2
	addps	xmm5, xmm0			; xmm5 += xmm0 * xmm2
	dec	rdx
	jz	.loop_end
	ALIGN 16
.loop_start:
	; start by reading the next sample
	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[sample]
	add	rcx, 4
	shufps	xmm0, xmm0, 0			; xmm0 = data[sample],data[sample],data[sample],data[sample]
	shufps	xmm2, xmm2, 93h			; 93h=2-1-0-3 => xmm2 gets rotated left by one float
	movss	xmm2, xmm0
	mulps	xmm0, xmm2			; xmm0 = xmm0 * xmm2
	addps	xmm5, xmm0			; xmm5 += xmm0 * xmm2
	dec	rdx
	jnz	.loop_start
.loop_end:
	; store autoc
	movups	[r9d], xmm5

.end:
	emms
	ret

	ALIGN 16
cident FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_8
	;   r9d    == autoc[]
	;   r8d    == lag
	;   rdx    == data_len
	;   rcx    == data[]

	;ASSERT(lag > 0)
	;ASSERT(lag <= 8)
	;ASSERT(lag <= data_len)

	;	for(coeff = 0; coeff < lag; coeff++)
	;		autoc[coeff] = 0.0;
	xorps	xmm4, xmm4
	xorps	xmm5, xmm5

	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[0]
	add	rcx, 4
	movaps	xmm2, xmm0			; xmm2 = 0,0,0,data[0]
	shufps	xmm0, xmm0, 0			; xmm0 == data[sample],data[sample],data[sample],data[sample] = data[0],data[0],data[0],data[0]
	movaps	xmm1, xmm0			; xmm1 == data[sample],data[sample],data[sample],data[sample] = data[0],data[0],data[0],data[0]
	xorps	xmm3, xmm3			; xmm3 = 0,0,0,0
.warmup:					; xmm3:xmm2 == data[sample-7],data[sample-6],...,data[sample]
	mulps	xmm0, xmm2
	mulps	xmm1, xmm3			; xmm1:xmm0 = xmm1:xmm0 * xmm3:xmm2
	addps	xmm4, xmm0
	addps	xmm5, xmm1			; xmm5:xmm4 += xmm1:xmm0 * xmm3:xmm2
	dec	rdx
	jz	.loop_end
	ALIGN 16
.loop_start:
	; start by reading the next sample
	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[sample]
	; here we reorder the instructions; see the (#) indexes for a logical order
	shufps	xmm2, xmm2, 93h			; (3) 93h=2-1-0-3 => xmm2 gets rotated left by one float
	add	rcx, 4				; (0)
	shufps	xmm3, xmm3, 93h			; (4) 93h=2-1-0-3 => xmm3 gets rotated left by one float
	shufps	xmm0, xmm0, 0			; (1) xmm0 = data[sample],data[sample],data[sample],data[sample]
	movss	xmm3, xmm2			; (5)
	movaps	xmm1, xmm0			; (2) xmm1 = data[sample],data[sample],data[sample],data[sample]
	movss	xmm2, xmm0			; (6)
	mulps	xmm1, xmm3			; (8)
	mulps	xmm0, xmm2			; (7) xmm1:xmm0 = xmm1:xmm0 * xmm3:xmm2
	addps	xmm5, xmm1			; (10)
	addps	xmm4, xmm0			; (9) xmm5:xmm4 += xmm1:xmm0 * xmm3:xmm2
	dec	rdx
	jnz	.loop_start
.loop_end:
	; store autoc
	movups	[r9d], xmm4
	movups	[r9d + 16], xmm5

.end:
	emms
	ret


	ALIGN 16
cident FLAC__lpc_compute_autocorrelation_asm_ia64_sse_lag_12
	;   r9d    == autoc[]
	;   r8d    == lag
	;   rdx    == data_len
	;   rcx    == data[]

	;ASSERT(lag > 0)
	;ASSERT(lag <= 12)
	;ASSERT(lag <= data_len)

	movups	[r9d], xmm6 ; save xmm6, which might be used by the caller
	movups	[r9d + 16], xmm7 ; save xmm7, which might be used by the caller

	;	for(coeff = 0; coeff < lag; coeff++)
	;		autoc[coeff] = 0.0;
	xorps	xmm5, xmm5
	xorps	xmm6, xmm6
	xorps	xmm7, xmm7

	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[0]
	add	rcx, 4
	movaps	xmm2, xmm0			; xmm2 = 0,0,0,data[0]
	shufps	xmm0, xmm0, 0			; xmm0 == data[sample],data[sample],data[sample],data[sample] = data[0],data[0],data[0],data[0]
	xorps	xmm3, xmm3			; xmm3 = 0,0,0,0
	xorps	xmm4, xmm4			; xmm4 = 0,0,0,0
.warmup:					; xmm3:xmm2 == data[sample-7],data[sample-6],...,data[sample]
	movaps	xmm1, xmm0
	mulps	xmm1, xmm2
	addps	xmm5, xmm1
	movaps	xmm1, xmm0
	mulps	xmm1, xmm3
	addps	xmm6, xmm1
	mulps	xmm0, xmm4
	addps	xmm7, xmm0			; xmm7:xmm6:xmm5 += xmm0:xmm0:xmm0 * xmm4:xmm3:xmm2
	dec	rdx
	jz	.loop_end
	ALIGN 16
.loop_start:
	; start by reading the next sample
	movss	xmm0, [rcx]			; xmm0 = 0,0,0,data[sample]
	add	rcx, 4
	shufps	xmm0, xmm0, 0			; xmm0 = data[sample],data[sample],data[sample],data[sample]

	; shift xmm4:xmm3:xmm2 left by one float
	shufps	xmm2, xmm2, 93h			; 93h=2-1-0-3 => xmm2 gets rotated left by one float
	shufps	xmm3, xmm3, 93h			; 93h=2-1-0-3 => xmm3 gets rotated left by one float
	shufps	xmm4, xmm4, 93h			; 93h=2-1-0-3 => xmm4 gets rotated left by one float
	movss	xmm4, xmm3
	movss	xmm3, xmm2
	movss	xmm2, xmm0

	; xmm7:xmm6:xmm5 += xmm0:xmm0:xmm0 * xmm3:xmm3:xmm2
	movaps	xmm1, xmm0
	mulps	xmm1, xmm2
	addps	xmm5, xmm1
	movaps	xmm1, xmm0
	mulps	xmm1, xmm3
	addps	xmm6, xmm1
	mulps	xmm0, xmm4
	addps	xmm7, xmm0

	dec	rdx
	jnz	.loop_start
.loop_end:
	; store autoc
	movups	[r9d + 32], xmm7
	movups	xmm7, [r9d + 16] ; restore xmm7
	movups	[r9d + 16], xmm6
	movups	xmm6, [r9d] ; restore xmm6
	movups	[r9d], xmm5

.end:
	emms
	ret


%ifdef OBJ_FORMAT_elf
       section .note.GNU-stack noalloc
%endif
