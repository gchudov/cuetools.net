// Reed-Solomon encoding/erasure decoding
// Fast version on GF(2^n_field) using Walsh transform

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
//

// ************************************************
// ************************************************

#include "reed_solomon.h"
#include "stdlib.h"
#include "stdio.h"

// ************************************************
// ************************************************

int n_field;
int n_walsh;
int N_field;
int N_walsh;
int modulo;

// GF tables
symbol *log_table;
symbol *exp_table;

// Encoding/Decoding product
symbol *product;
symbol *product_enc;

// Used to compute product
symbol *log_walsh;
symbol *pos;

// extra memory needed for fast evaluation
symbol *coeff, *t_coeff;
symbol *inverse, *t_inverse;

void memory_init()
{
    log_table = (symbol *)malloc(sizeof(symbol)*N_field);
    exp_table = (symbol *)malloc(sizeof(symbol)*2*N_field);

    product = (symbol *)malloc(sizeof(symbol)*N_walsh);
    product_enc  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    log_walsh = (symbol *)malloc(sizeof(symbol)*N_walsh);
    pos  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    inverse = t_inverse = (symbol *) malloc(sizeof(symbol) * N_walsh * n_field + 8);
    coeff = t_coeff = (symbol *) malloc(sizeof(symbol) * N_walsh * n_field + 8);

    // allign memory for SSE operation
    while ((int)(inverse) & 0xf) inverse++;
    while ((int)(coeff) & 0xf) coeff++;
}

void memory_clear()
{
      free(log_table);
      free(exp_table);
      free(product);
      free(product_enc);
      free(log_walsh);
      free(pos);
      free(t_inverse);
      free(t_coeff);
}


// ************************************************
// ************************************************

// compute the tables for the finite field operations
// exp_table is twice the needed size so we do not need to
// perform a modulo each time.

// list of some primitive polynomials
int primitive[] = {
    0,1,6,
    0,3,7,
    0,2,3,4,8,
    0,4,9,
    0,3,10,
    0,2,11,
    0,1,4,6,12,
    0,1,3,4,13,
    0,1,11,12,14,
    0,1,15,
    0,1,3,12,16,
    0,3,17,
    0,3,20,
    0,3,25,
    100         // fallback
};

// will be set to the primitive poly of the field
// used only by the fast algorithm version
int *poly;
int  weight;

void fill_table()
{
    int i;

    // put the primitive poly in mask
    int temp=0;
    int mask=0;
    int index=0;
    while (1) {
        if (primitive[index]==0) {
            mask=0;
            temp=index;
        }
        mask ^= 1 << primitive[index];
        if (primitive[index]>=n_field) break;
        index++;
    }

    // used for the fast version only
    poly = &primitive[temp];
    weight = index-temp;

    if (primitive[index]!=n_field) {
         printf("primitive poly for GF %d not found !\n", n_field);
    }

    // clock the lfsr (multiply by X)
    // fill the table
    int state=1;
    for (i=0; i<modulo; i++)
    {
        if (log_table[state]!=0) {
            printf("polynomial is not primitive\n");
        }

        log_table[state]=i;
        exp_table[i]=state;
        exp_table[modulo+i]=state;

        state <<=1;
        if (state>>n_field) state^=mask;
        if (state>>n_field!=0) exit(0);
    }

    // small change so we have bijection
    log_table[0]=0;
    log_table[1]=modulo;
    exp_table[2*modulo]=1;
}

// ************************************************
// ************************************************

void walsh_mod(symbol *vect, int size);
void compute_product();
void walsh_field(symbol *vect);

// ************************************************
// ************************************************

void log_walsh_init()
{
    int i;
    for (i=0; i<N_walsh; i++)
        log_walsh[i] = log_table[i];
    log_walsh[1]=0;
    walsh_mod(log_walsh, N_walsh);

    // multiply it by the correct value,
    // so computation are easier later
    int shift = n_field - n_walsh;
    for (i=0; i<N_walsh; i++)
        log_walsh[i] = ((int) log_walsh[i] << shift) % modulo;
}

void fast_init()
{
    int i,j;

    // clear coeff
    // It need to be done only once
    for (i=0; i<N_walsh*n_field; i++)
        coeff[i]=0;

    // fill the inverse table
    for (i=0; i<N_walsh; i++) {
        int t = exp_table[(modulo-log_table[i]) % modulo];
        if(i==0) t=0;
        for (j=0; j<n_field; j++)
            inverse[i*n_field + j] = (t>>j)&1;
    }

    // precompute the Walsh transforms of the inverse
    walsh_field(inverse);

    // everything is even if n_walsh=16
    // divide by two so computation fit on 16 bits!
    // otherwise, multiply by power of two, so
    // shift factor is always 15.
    if (n_walsh==16)
        for (i=0; i<n_field*N_walsh; i++) inverse[i] >>= 1;
    else {
        for (i=0; i<n_field*N_walsh; i++) inverse[i] <<= (15-n_walsh);
    }
}


// *******************************************************
// *******************************************************

void code_init(int nf, int nw)
{
    if (nf>31 || nw > nf) {
        printf("incorrect field parameters\n");
        exit(0);
    }

    n_field = nf;
    n_walsh = nw;
    N_field = 1<<n_field;
    N_walsh = 1<<n_walsh;
    modulo = N_field-1;

    memory_init();
    fill_table();
    log_walsh_init();
    fast_init();
}

void code_clear()
{
    memory_clear();
}

// *******************************************************
// *******************************************************

// for encoding, we can precompute the product once
void encode_init(int K)
{
    int i;

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[i]=1;

    // compute product
    compute_product();

    // save it in product_enc
    // so it is not overwritten by any decoding
    for (i=0; i<N_walsh; i++)
        product_enc[i] = product[i];
}

// ************************************************
// ************************************************

// this can be optimized a bit...
// But it is not critical at all.

// Perform a Walsh transform and keep the coeffs mod (modulo)
// The transformation is involutive if n_walsh = n_field.
void walsh_mod(symbol *vect, int size)
{
    int i,j,step;
    step=1;
    while (step<size) {
        i=0;
        while (i<size) {
            j = step;
            while (j--)
            {
                int t=vect[i];
                int b=vect[i+step];
                int a=t+b;

                b = t + modulo - b;

                // modulo 2^m - 1
                // a = a_1 2^m + a_0
                // and 2^m = 1 mod (2^m - 1)
                // remark : result can be 2^m-1 = 0

//                a = (a & modulo) + (a>>n_field);
//                b = (b & modulo) + (b>>n_field);

                // on GF 2^16
                a = a + (a>>16);
                b = b + (b>>16);

                vect[i]=a;
                vect[i+step]=b;

                i++;
            }
            i+=step;
        }
        step<<=1;
    }
}

// ************************************************
// ************************************************

// compute the product (3) of the paper
// return in product the logarithm of the product
void compute_product()
{
    int i;
    // initialisation
    for (i=0; i<N_walsh; i++)
        product[i]=pos[i];

    // Walsh transform
    walsh_mod(product,N_walsh);

    // multiplication
    // need long long here if n_field > 16
    // otherwise int is ok.
    for (i=0; i<N_walsh; i++) {
        product[i] = ((unsigned int)product[i] * (unsigned int)log_walsh[i]) % modulo;
    }

    // inverse Walsh transform
    // it is not involutive if n_field != n_walsh,
    // But we multiplied the log_walsh by (1<<(n_field-n_walsh))
    // so that it works anyway !
    walsh_mod(product,N_walsh);
}

// Same but quadratic version
// Here only for debuging purpose
void compute_product_quadratic(int K, int *positions)
{
    int i,j;

    for (j=0; j<N_walsh; j++)
        product[j] = log_table[j ^ positions[0]];

    for (i=1; i<K; i++) {
        for (j=0; j<N_walsh; j++) {
            int t = product[j] + log_table[j ^ positions[i]];
            if (t>modulo) t-= modulo;
            product[j] = t;
        }
    }
}

// *******************************************************
// *******************************************************

// Perform a Walsh transform
// Nothing special this time, except that we do not
// have to worry about any overflow since in our use,
// only the bit n_walsh will be important
//
// special version that do n_field transforms at once.
// one of the limiting part in the complexity

void walsh_field_generic(symbol *vect)
{
    int i,j;
    int step=n_field;
    int size=n_field * N_walsh;
    while (step<size)
    {
        symbol *p=vect;
        symbol *q=vect+step;
        i=size/(2*step);
        while (i--) {
            j=step;
            while (j--) {
                symbol a = *p;
                symbol b = *q;
                *(p++) = a + b;
                *(q++) = a - b;
            }
            p+=step;
            q+=step;
        }
        step<<=1;
    }
}

// In some critical part of the code, we gain a lot
// using a constant instead of n_field.
#define NFIELD 16

#ifdef SSE

// Perform a Walsh transform on 4 expanded field element
// each taking 16 x 16bits
void walsh_end(symbol *p) {
    __asm__ __volatile__ (
        "movdqa   (%%esi), %%xmm0\n"
        "movdqa 16(%%esi), %%xmm1\n"
        "movdqa 32(%%esi), %%xmm2\n"
        "movdqa 48(%%esi), %%xmm3\n"
        "movdqa 64(%%esi), %%xmm4\n"
        "movdqa 80(%%esi), %%xmm5\n"
        "movdqa 96(%%esi), %%xmm6\n"
        "movdqa 112(%%esi), %%xmm7\n"

        "psubw    %%xmm2, %%xmm0\n"
        "psubw    %%xmm3, %%xmm1\n"
        "psubw    %%xmm6, %%xmm4\n"
        "psubw    %%xmm7, %%xmm5\n"

        "psllw   $1,%%xmm2\n"
        "psllw   $1,%%xmm3\n"
        "psllw   $1,%%xmm6\n"
        "psllw   $1,%%xmm7\n"

        "paddw    %%xmm0, %%xmm2\n"
        "paddw    %%xmm1, %%xmm3\n"
        "paddw    %%xmm4, %%xmm6\n"
        "paddw    %%xmm5, %%xmm7\n"

        "psubw    %%xmm4, %%xmm0\n"
        "psubw    %%xmm5, %%xmm1\n"
        "psubw    %%xmm6, %%xmm2\n"
        "psubw    %%xmm7, %%xmm3\n"

        "psllw   $1,%%xmm4\n"
        "psllw   $1,%%xmm5\n"
        "psllw   $1,%%xmm6\n"
        "psllw   $1,%%xmm7\n"

        "paddw    %%xmm0, %%xmm4\n"
        "paddw    %%xmm1, %%xmm5\n"
        "paddw    %%xmm2, %%xmm6\n"
        "paddw    %%xmm3, %%xmm7\n"

        "movdqa   %%xmm6,   (%%esi)\n"
        "movdqa   %%xmm7, 16(%%esi)\n"
        "movdqa   %%xmm4, 32(%%esi)\n"
        "movdqa   %%xmm5, 48(%%esi)\n"
        "movdqa   %%xmm2, 64(%%esi)\n"
        "movdqa   %%xmm3, 80(%%esi)\n"
        "movdqa   %%xmm0, 96(%%esi)\n"
        "movdqa   %%xmm1, 112(%%esi)\n"
    : : "S"(p) : "memory");
}

// Perform a Walsh step (A,B) -> (A+B, A-B)
// where sizeof A = sizeof B = s*32 bits
void walsh_step(symbol *p, symbol *q, int s) {
    while (s--) {
        __asm__ __volatile__ (
            "movdqa   (%%esi), %%xmm0\n"
            "movdqa 16(%%esi), %%xmm1\n"
            "movdqa 32(%%esi), %%xmm2\n"
            "movdqa 48(%%esi), %%xmm3\n"

            "movdqa   (%%edi), %%xmm4\n"
            "movdqa 16(%%edi), %%xmm5\n"
            "movdqa 32(%%edi), %%xmm6\n"
            "movdqa 48(%%edi), %%xmm7\n"

            "psubw    %%xmm4, %%xmm0\n"
            "psubw    %%xmm5, %%xmm1\n"
            "psubw    %%xmm6, %%xmm2\n"
            "psubw    %%xmm7, %%xmm3\n"

            "psllw   $1,%%xmm4\n"
            "psllw   $1,%%xmm5\n"
            "psllw   $1,%%xmm6\n"
            "psllw   $1,%%xmm7\n"

            "paddw    %%xmm0, %%xmm4\n"
            "paddw    %%xmm1, %%xmm5\n"
            "paddw    %%xmm2, %%xmm6\n"
            "paddw    %%xmm3, %%xmm7\n"

            "movdqa   %%xmm4,   (%%esi)\n"
            "movdqa   %%xmm5, 16(%%esi)\n"
            "movdqa   %%xmm6, 32(%%esi)\n"
            "movdqa   %%xmm7, 48(%%esi)\n"

            "movdqa   %%xmm0,   (%%edi)\n"
            "movdqa   %%xmm1, 16(%%edi)\n"
            "movdqa   %%xmm2, 32(%%edi)\n"
            "movdqa   %%xmm3, 48(%%edi)\n"
        : : "S"(p), "D"(q) : "memory");

        p += 32;
        q += 32;
    }
}

// Perform a Walsh step (A,B) -> (A+B, A-B)
// where sizeof A = sizeof B = s*16 bits
void walsh_step_simple(symbol *p, symbol *q, int s) {
    while (s--) {
        __asm__ __volatile__ (
            "movdqa   (%%esi), %%xmm0\n"
            "movdqa 16(%%esi), %%xmm1\n"

            "movdqa   (%%edi), %%xmm4\n"
            "movdqa 16(%%edi), %%xmm5\n"

            "psubw    %%xmm4, %%xmm0\n"
            "psubw    %%xmm5, %%xmm1\n"

            "psllw   $1,%%xmm4\n"
            "psllw   $1,%%xmm5\n"

            "paddw    %%xmm0, %%xmm4\n"
            "paddw    %%xmm1, %%xmm5\n"

            "movdqa   %%xmm4,   (%%esi)\n"
            "movdqa   %%xmm5, 16(%%esi)\n"

            "movdqa   %%xmm0,   (%%edi)\n"
            "movdqa   %%xmm1, 16(%%edi)\n"
        : : "S"(p), "D"(q) : "memory");

        p += 16;
        q += 16;
    }
}

// iterative DFS
void walsh_field_iter(symbol *p, int size)
{
    int i;
    size/=4;
    for (i=0; i<size; i++) {
        // start of A
        symbol *q = p;

        // size of the Walsh transform in 16bits
        int taille=2;

        // first Walsh transform
        // alternative with size/=2: walsh_step_simple(q, q + NFIELD*taille, taille);
        walsh_end(q);

        // More step depending on the trailing zero in the binary
        // representation of i+1
        int t=i+1;
        while ((t&1)==0) {
            taille <<= 1;
            q -= NFIELD*taille;
            walsh_step(q, q + NFIELD*taille, taille/2);
            t >>= 1;
        };
        // next bloc
        p += 4*NFIELD;
    }
}

// Recursive version
void walsh_field_rec(symbol *p, int size)
{
//  if (size==2) return walsh_step_simple(p,p+n_field,1);
    if (size==4) return walsh_end(p);

    size /= 2;
    symbol *q = p + size*NFIELD;
    walsh_field_rec(p,size);
    walsh_field_rec(q,size);

//  walsh_step_simple(p,q,size);
    walsh_step(p,q,size/2);
}

void walsh_field(symbol *vect)
{
    return walsh_field_iter(vect, N_walsh);
    return walsh_field_rec(vect, N_walsh);
    int i,j;
    int step=n_field;
    int size=n_field * N_walsh;
    while (step<size)
    {
        symbol *p=vect;
        symbol *q=vect+step;
        i=size/(2*step);
        while (i--) {
            j=step/16;
            while (j--) {
                __asm__ __volatile__ (
                    "movdqa   (%%esi), %%xmm0\n"
                    "movdqa 16(%%esi), %%xmm1\n"

                    "movdqa   (%%edi), %%xmm2\n"
                    "movdqa 16(%%edi), %%xmm3\n"

                    "movdqa   %%xmm0, %%xmm4\n"
                    "movdqa   %%xmm1, %%xmm5\n"

                    "paddw    %%xmm2, %%xmm0\n"
                    "paddw    %%xmm3, %%xmm1\n"

                    "psubw    %%xmm2, %%xmm4\n"
                    "psubw    %%xmm3, %%xmm5\n"

                    "movdqa   %%xmm0,   (%%esi)\n"
                    "movdqa   %%xmm1, 16(%%esi)\n"

                    "movdqa   %%xmm4,   (%%edi)\n"
                    "movdqa   %%xmm5, 16(%%edi)\n"
                : : "S"(p), "D"(q) : "memory");

                p += 16;
                q += 16;
            }
            p+=step;
            q+=step;
        }
        step<<=1;
    }
}


#else

void walsh_field(symbol *vect) {
    walsh_field_generic(vect);
}

#endif

// *******************************************************
// *******************************************************

// compute field product of the vector coeff and inverse
// for each [n_field] consecutive symbols of [inverse] and [coeff]
// multiply them as if they were element of the field in the
// standard basis.
//
// the difference is instead of coefficients in GF_2,
// the coefficients are in Z ...
//
// The other limiting part in the complexity (with walsh_field).
// The multiplication is quadratic in n_field
// and is performed N_walsh time.


// work with any field using n_field but slow
// look at the one on GF(2^16) for a fast version
void field_product_generic()
{
    int i,j,k;
    symbol s[64];

    symbol *inv = inverse;

    for (k=0; k<N_walsh; k++)
    {
        symbol *c = &coeff[k*n_field];
        symbol *base = &s[n_field];

        // put [c] in [s]
        for (i=0; i<n_field; i++) {
            base[i] = c[i];
            c[i]=0;
            s[i]=0;
        }

        // product
        for (i=0; i<n_field; i++)
        {
            // add inv[i]*[s] to [c]
            symbol t = *(inv++);
            for (j=0; j<n_field; j++) {
                c[j] += t * base[j];
            }
            // multiply [s] by x
            base--;
            t = base[n_field];
            for (j=0; j<weight; j++) {
                base[poly[j]] += t;
            }
        }
    }
}

#ifdef SSE

inline ssse_1()
{
    __asm__ __volatile__ (
        // multiply [base] by x

        // left shift of xmm3,xmm2
        // word out is in xmm7
        "movapd   %%xmm2, %%xmm6\n"
        "movapd   %%xmm3, %%xmm7\n"
        "pslldq   $2,%%xmm2\n"
        "pslldq   $2,%%xmm3\n"
        "psrldq   $14,%%xmm6\n"
        "psrldq   $14,%%xmm7\n"
        "pxor     %%xmm6,%%xmm3\n"

        // expand xmm7 to pos 0, 1, 3
        // add it to xmm2
        "pshuflw  $16,%%xmm7,%%xmm6\n"
        "paddw    %%xmm6,%%xmm2\n"

        // put value in pos 4 (8+4)
        // add it to xmm3
        // "psrldq  $6,%%xmm7\n"
        "pslldq  $8,%%xmm7\n"
        "paddw   %%xmm7,%%xmm3\n"

        // next inv ...
        "psrldq   $2, %%xmm4\n"

        // copy 8 times low word in xmm4
        "pshuflw  $0, %%xmm4, %%xmm6\n"
        "pshufd   $0, %%xmm6, %%xmm6\n"
        "movapd   %%xmm6, %%xmm7\n"

        // multiply constant by base
        "pmullw   %%xmm2, %%xmm6\n"
        "pmullw   %%xmm3, %%xmm7\n"

        // add it to [c]
        "paddw    %%xmm6, %%xmm0\n"
        "paddw    %%xmm7, %%xmm1\n"
    : );
}

inline ssse_2()
{
    __asm__ __volatile__ (
        // multiply [base] by x

        // left shift of xmm3,xmm2
        // word out is in xmm7
        "movapd   %%xmm2, %%xmm6\n"
        "movapd   %%xmm3, %%xmm7\n"
        "pslldq   $2,%%xmm2\n"
        "pslldq   $2,%%xmm3\n"
        "psrldq   $14,%%xmm6\n"
        "psrldq   $14,%%xmm7\n"
        "pxor     %%xmm6,%%xmm3\n"

        // expand xmm7 to pos 0, 1, 3
        // add it to xmm2
        "pshuflw  $16,%%xmm7,%%xmm6\n"
        "paddw    %%xmm6,%%xmm2\n"

        // put value in pos 4 (8+4=12)
        // add it to xmm3
        // "psrldq  $6,%%xmm7\n"
        "pslldq  $8,%%xmm7\n"
        "paddw   %%xmm7,%%xmm3\n"

        // copy 8 times low word in xmm5
        "pshuflw  $0, %%xmm5, %%xmm6\n"
        "pshufd   $0, %%xmm6, %%xmm6\n"
        "movapd   %%xmm6, %%xmm7\n"

        // next inv ...
        "psrldq   $2, %%xmm5\n"

        // multiply constant by base
        "pmullw   %%xmm2, %%xmm6\n"
        "pmullw   %%xmm3, %%xmm7\n"

        // add it to [c]
        "paddw    %%xmm6, %%xmm0\n"
        "paddw    %%xmm7, %%xmm1\n"
    : );
}

inline sse_loop()
{
    int i;
    for (i=1; i<NFIELD; i++)
    {
        __asm__ __volatile__ (
            // multiply [base] by x

            // left shift of xmm3,xmm2
            // word out is in xmm7
            "movapd   %%xmm2, %%xmm6\n"
            "movapd   %%xmm3, %%xmm7\n"
            "pslldq   $2,%%xmm2\n"
            "pslldq   $2,%%xmm3\n"
            "psrldq   $14,%%xmm6\n"
            "psrldq   $14,%%xmm7\n"
            "pxor     %%xmm6,%%xmm3\n"

            // expand xmm7 to pos 0, 1, 3
            // add it to xmm2
            "pshuflw  $16,%%xmm7,%%xmm6\n"
            "paddw    %%xmm6,%%xmm2\n"

            // put value in pos 4 (8+4)
            // add it to xmm3
            // "psrldq  $6,%%xmm7\n"
            "pslldq  $8,%%xmm7\n"
            "paddw   %%xmm7,%%xmm3\n"

            // next inv ...
            "psrldq   $2, %%xmm4\n"
            "movapd   %%xmm5, %%xmm6\n"
            "pslldq   $14, %%xmm6\n"
            "pxor     %%xmm6, %%xmm4\n"
            "psrldq   $2, %%xmm5\n"

            // copy 8 times low word in xmm4
            "pshuflw  $0, %%xmm4, %%xmm6\n"
            "pshufd   $0, %%xmm6, %%xmm6\n"
            "movapd   %%xmm6, %%xmm7\n"

            // multiply constant by base
            "pmullw   %%xmm2, %%xmm6\n"
            "pmullw   %%xmm3, %%xmm7\n"

            // add it to [c]
            "paddw    %%xmm6, %%xmm0\n"
            "paddw    %%xmm7, %%xmm1\n"
        : );
    }
}

void field_product_16()
{
    int i,k;

    symbol *inv = inverse;
    symbol *c = coeff;

    for (k=0; k<N_walsh; k++)
    {
        __asm__ __volatile__ (
            // load c into xmm0,xmm1
            "movdqa   (%%edi), %%xmm0\n"
            "movdqa 16(%%edi), %%xmm1\n"

            // init base into xmm2,xmm3
            "movdqa   %%xmm0, %%xmm2\n"
            "movdqa   %%xmm1, %%xmm3\n"

            // load inv into xmm4,xmm5
            "movdqa   (%%esi), %%xmm4\n"
            "movdqa 16(%%esi), %%xmm5\n"

            // make 8 copy of inv[0]
            "pshuflw  $0, %%xmm4, %%xmm6\n"
            "pshufd   $0, %%xmm6, %%xmm6\n"

            // multiply [c] by constant
            "pmullw   %%xmm6, %%xmm0\n"
            "pmullw   %%xmm6, %%xmm1\n"

        : : "S"(inv), "D"(c) : "memory");

        // product with loop
//        sse_loop();

        // product without loop
        // the last one can be simplified a bit,but really small gain.
        ssse_1();ssse_1();ssse_1();ssse_1();ssse_1();ssse_1();ssse_1();
        ssse_2();ssse_2();ssse_2();ssse_2();ssse_2();ssse_2();ssse_2();ssse_2();

        // put result back in [c]
        __asm__ __volatile__ (
            "movdqa   %%xmm0,   (%%edi)\n"
            "movdqa   %%xmm1, 16(%%edi)\n"
        : : "D"(c) : "memory");

        // next c
        c += NFIELD;
        inv += NFIELD;
    }
}

#else

void field_product_16()
{
    int i,j,k;
    symbol s[32];

    symbol *inv = inverse;
    symbol *c = coeff;
    symbol t;

    for (k=0; k<N_walsh; k++)
    {
        symbol *base = s + NFIELD;

        // put [c] in [base]
        // multiply [c] by inv[0]
        t = *(inv++);
        for (i=0; i<NFIELD; i++) {
            base[i] = c[i];
            c[i] *=t;
        }

        // product
        for (i=1; i<NFIELD; i++)
        {
            // multiply [base] by x
            base--;
            t = base[0] = base[NFIELD];
            base[1] += t;
            base[3] += t;
            base[12] += t;

            // add inv[i]*[base] to [c]
            t = *(inv++);
            for (j=0; j<NFIELD; j++) {
                c[j] += t * base[j];
            }
        }

        // next c
        c += NFIELD;
    }
}

#endif

void (* field_product)() = field_product_16;


// *******************************************************
// *******************************************************

// In the case where k and n-k are smaller than or equal to 2^i
// We can gain a factor 2 for encoding, by doing the convolution on
// length 2^i rather than 2^{i+1} ...
// Not implemented here.
// Require to change the parity positions ...

void fast_encode(int N, int K, int S, symbol *data, symbol *parity)
{
    int i,j,k;

    // no need to clear coeff afterwards
    // because after the second walsh transform, the
    // least significant bits are 0 anyway...
    // memset(coeff, 0, N_walsh*n_field*sizeof(symbol));

    // encode codeword by codeword
    for (j=0; j<S; j++)
    {
        // put received symbols in places
        symbol *temp = coeff;
        for (i=0; i<K; i++)
        {
            int t=data[i*S+j];
            if (t) t=exp_table[log_table[t] + modulo - product_enc[i]];

            for (k=0; k<NFIELD; k++) {
                *(temp++) = t;
                t >>=1;
                // we are interested only in the parity
                // so no need to add &1;
            }
        }

        // compute the Walsh transforms of the coefficient
        walsh_field(coeff);

        // multiply with precomputed Walsh transform of the inverse
        field_product();

        // walsh transform again
        walsh_field(coeff);

        // put decoded symbol in place
        // final multiplication by the product
        temp = coeff + K*n_field;
        for (i=0; i<N-K; i++) {
            // reconstruct the field element from
            // its shift coordinates
            int t=*(temp++);
            for (k=1; k<NFIELD; k++) {
                t >>= 1;
                t ^= *(temp++);
            }

            if (t) t = exp_table[log_table[t] + product_enc[K+i]];
            parity[i*S + j] = t;
        }
    }
}

void fast_decode(int K, int S, int *positions, symbol *data, symbol *packets)
{
    int i,j,k;

    // copy the systematic pieces in place
    for (i=0; i<K; i++)
        if (positions[i]<K)
            for (j=0; j<S; j++)
                packets[positions[i]*S + j] = data[i*S +j];

    // precomputation
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[positions[i]]=1;
    compute_product();

    // decode codeword by codeword
    for (j=0; j<S; j++)
    {
        // put received symbols in places
        for (i=0; i<K; i++)
        {
            symbol *temp = coeff + positions[i]*NFIELD;
            int t=data[i*S+j];
            if (t)
                t=exp_table[log_table[t] + modulo - product[positions[i]]];

            for (k=0; k<NFIELD; k++) {
                *(temp++) = t;
                t >>=1;
                // we are interested only in the parity
                // so no need to add &1;
            }
        }

        // compute the Walsh transforms of the coefficient
        walsh_field(coeff);

        // multiply with precomputed Walsh transform of the inverse
        field_product();

        // walsh transform again
        walsh_field(coeff);

        // put decoded symbol in place
        // final multiplication by the product
        for (i=0; i<K; i++) if (pos[i]==0)
        {
            symbol *temp = coeff + i*NFIELD;

            // reconstruct the field element from
            // its shift coordinates
            // only the 15 bits of *temp is non-zero
            int t=*(temp++);
            for (k=1; k<NFIELD; k++) {
                t >>= 1;
                t ^= *(temp++);
            }

            if (t) t = exp_table[log_table[t] + product[i]];
            packets[i*S + j] = t;
        }
    }
}

