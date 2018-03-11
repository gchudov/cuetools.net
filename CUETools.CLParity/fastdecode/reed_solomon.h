// Reed-Solomon encoding/erasure decoding

// Implementation of the algorithms described in
// Efficient erasure decoding of Reed-Solomon codes
// http://arxiv.org/abs/0901.1886v1

// (c) 2009 Frederic didier.
// Any feedback is very welcome. For any question, comments,
// see http://algo.epfl.ch/~didier/reed_solomon.html or email
// frederic.didier@epfl.ch

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials
//    provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.


#include "stdlib.h"
#include "string.h"
#include "stdint.h"
#include "stdio.h"

// type used to store one field symbol
// With short int, we can work up to GF(16)
// and we can apply the fast algo up to N_walsh=15
// If one wants to work on bigger fied, replace this by int.
typedef uint16_t symbol;
typedef uint8_t byte;

// Common initialisation functions
extern void fill_table(int n_field);
extern void code_init(int n_walsh);
extern void code_clear();

// quadratic algo
extern void encode_init(int N, int K);

extern void (*process)(int, void *, void *, int);
extern void (*process_eq)(int, void *, void *, int);

extern void use_xor();
extern void use_xor2();
extern void use_table();
extern void use_direct();

extern void use_quadratic();
extern void use_incremental();
extern void use_karatsuba();
extern void use_special();

extern void (*RS_encode)(int N, int K, int S, void *info, void *output);
extern void (*RS_decode)(int N, int K, int S, int *pos, void *received, void *output);

