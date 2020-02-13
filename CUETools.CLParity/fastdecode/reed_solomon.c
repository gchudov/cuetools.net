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
//

// ************************************************
// ************************************************

#include "reed_solomon.h"

// ************************************************
// ************************************************
int n_field;
int N_field;
int modulo;

symbol *log_table;
symbol *exp_table;


uint8_t *mult_table;

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

// contain the primitive poly in binary form
// used in xor type field multiplication
int  field_mask;

// init field.
void fill_table(int nf)
{
    n_field = nf;
    N_field = 1<<n_field;
    modulo = N_field-1;

    log_table = (symbol *)malloc(sizeof(symbol)*N_field);
    exp_table = (symbol *)malloc(sizeof(symbol)*2*N_field);

    // put the primitive poly in mask
    int temp=0;
    int pos=0;
    field_mask = 0;
    while (1) {
        if (primitive[pos]==0) {
            field_mask=0;
            temp=pos;
        }
        field_mask ^= 1 << primitive[pos];
        if (primitive[pos]>=n_field) break;
        pos++;
    }

    // used for the fast version only
    poly = &primitive[temp];
    weight = pos-temp;

    if (primitive[pos]!=n_field) {
         printf("primitive poly for GF %d not found !\n", n_field);
    }

    // clock the lfsr (multiply by X)
    int state=1;
    int i;
    for (i=0; i<modulo; i++)
    {
        if (log_table[state]!=0) {
            printf("polynomial is not primitive\n");
        }

        log_table[state]=i;
        exp_table[i]=state;
        exp_table[modulo+i]=state;

        state <<=1;
        if (state>>n_field) state^=field_mask;
        if (state>>n_field!=0) exit(0);
    }

    // useful since later
    // since log_table[0]=0
    // we set log_table[1]=modulo
    // so log_table is a bijection...
    log_table[0]=0;
    log_table[1]=modulo;
    exp_table[2*modulo]=1;

    // for GF(2^8)
    if (n_field==8) {
        int i,j;
        mult_table = (uint8_t *) malloc(256*256);
        for (i=0; i<256; i++)
        for (j=0; j<256; j++) {
            if (j==0) mult_table[i*256+j]=0;
            else mult_table[i*256+j]=exp_table[i+log_table[j]];
        }
    }
}

// *******************************************************
// *******************************************************

void packet_clear(void *p, int S)
{
    memset(p, 0, S);
}

void packet_log(void *p, int S)
{
    int i;
    uint8_t *pt = (uint8_t *)p;
    for (i=0; i<S; i++) {
        *pt = log_table[*pt];
        pt++;
    }
}

void packet_log16(void *p, int S)
{
    int i;
    uint16_t *pt = (uint16_t *)p;
    for (i=0; i<S/2; i++) {
        *pt = log_table[*pt];
        pt++;
    }
}

// *******************************************************
// *******************************************************

#define TABLE_INIT \
    symbol *table = &exp_table[log_cte]; \
    int t = table[0]; \
    table[0]=0;

#define TABLE_END \
    table[0]=t;

#define USE(a) \
    a *src = (a *)p_src; \
    a *dst = (a *)p_dst; \
    S /= sizeof(a);

#define LOOP(a) \
    int i; \
    for (i=0; i<S; i++) (a);

void memxor(void *p_dst, void* p_src, int S)
{
    USE(uint32_t);
    LOOP(*dst++ ^= *src++);
}

void process_packet_test8(int log_cte, void *p_dst, void *p_src, int S)
{
//    memcpy(p_dst, p_src, S);return;
    uint8_t *table = &mult_table[log_cte*256];
    USE(uint8_t);
    LOOP(*dst++ ^= table[*src++]);
}

void process_packet_test8_eq(int log_cte, void *p_dst, void *p_src, int S)
{
    uint8_t *table = &mult_table[log_cte*256];
    USE(uint8_t);
    LOOP(*dst++ = table[*src++]);
}

void process_packet_test(int log_cte, void *p_dst, void *p_src, int S)
{
    TABLE_INIT;
    USE(uint8_t);
    LOOP(*dst++ ^= table[log_table[*src++]]);
    TABLE_END;
}

void process_packet_test16(int log_cte, void *p_dst, void *p_src, int S)
{
    TABLE_INIT;
    USE(uint16_t);
    LOOP(*dst++ ^= table[log_table[*src++]]);
    TABLE_END;
}

void process_packet_test_eq(int log_cte, void *p_dst, void *p_src, int S)
{
    TABLE_INIT;
    USE(uint8_t);
    LOOP(*dst++ = table[log_table[*src++]]);
    TABLE_END;
}

void process_packet_test16_eq(int log_cte, void *p_dst, void *p_src, int S)
{
    TABLE_INIT;
    USE(uint16_t);
    LOOP(*dst++ = table[log_table[*src++]]);
    TABLE_END;
}

// *******************************************************
// *******************************************************

// process packet do the operation [symbol ^= symbol * cte]
// on all the symbols of a packet.
// [S] is the number of byte in one packet

// seg_size being size_t or int change the perf a lot ???

void process_packet_xor(int log_cte, void *p_dst, void *p_src, int S)
{
    int i,j,k;

    int seg_size = S / n_field;
    int cte = exp_table[log_cte];

    for (i=0; i<n_field; i++)
    {
        for (j=0; j<n_field; j++)
        {
            if ((cte >> j)&1) {
                uint32_t *src = (uint32_t *) (p_src + i*seg_size);
                uint32_t *dst = (uint32_t *) (p_dst + j*seg_size);

                for (k=0; k<seg_size/4; k++) {
                    *dst++ ^= *src++;
                }
            }
        }

        // multiply cte by X
        cte <<=1;
        if (cte>>n_field)
            cte^=field_mask;
    }
}

// less efficient even with inline ??
void process_packet_xor2(int log_cte, void *p_dst, void *p_src, int S)
{
    int i,j;

    int seg_size = S / n_field;
    int cte = exp_table[log_cte];

    for (i=0; i<n_field; i++)
    {
        for (j=0; j<n_field; j++)
        {
            if ((cte >> j)&1) {
                memxor(p_dst+j*seg_size, p_src+i*seg_size, seg_size);
            }
        }

        // multiply cte by X
        cte <<=1;
        if (cte>>n_field)
            cte^=field_mask;
    }
}

void process_packet_xor_eq(int log_cte, void *p_dst, void *p_src, int S)
{
    memset(p_dst, 0, S);
    process_packet_xor(log_cte, p_dst, p_src, S);
}

// *******************************************************
// *******************************************************

int multiply(int a, int b)
{
    int i;
    int res=0;
    for (i=0; i<n_field; i++) {
        if (b&1) res^= a;
        a <<=1;
        if (a >> n_field)
            a^=field_mask;
        b >>=1;
    }
    return res;
}

void process_packet_direct_simple(int log_cte, void *p_dst, void *p_src, int S)
{
    int i;
    int cte = exp_table[log_cte];
    
    uint16_t *src = (uint16_t *)p_src;
    uint16_t *dst = (uint16_t *)p_dst;
    for (i=0; i<S/2; i++)
    {
        int a=*(src++);
        *dst++ ^= multiply(a, cte);
    }
}

void process_packet_direct16(int log_cte, void *p_dst, void *p_src, int S)
{
    int i;
    int cte = exp_table[log_cte];
    int table[16];
    for (i=0;i<n_field;i++) {
        table[i]=cte;
        cte <<=1;
        if (cte >> n_field)
            cte^=field_mask;
    }

    uint16_t *src = (uint16_t *)p_src;
    uint16_t *dst = (uint16_t *)p_dst;
    for (i=0; i<S/2; i++)
    {
        int a=*(src++);
        int res;
        res  = (-((a>>0)&1)) & table[0];
        res ^= (-((a>>1)&1)) & table[1];
        res ^= (-((a>>2)&1)) & table[2];
        res ^= (-((a>>3)&1)) & table[3];
        res ^= (-((a>>4)&1)) & table[4];
        res ^= (-((a>>5)&1)) & table[5];
        res ^= (-((a>>6)&1)) & table[6];
        res ^= (-((a>>7)&1)) & table[7];
        res ^= (-((a>>8)&1)) & table[8];
        res ^= (-((a>>9)&1)) & table[9];
        res ^= (-((a>>10)&1)) & table[10];
        res ^= (-((a>>11)&1)) & table[11];
        res ^= (-((a>>12)&1)) & table[12];
        res ^= (-((a>>13)&1)) & table[13];
        res ^= (-((a>>14)&1)) & table[14];
        res ^= (-((a>>15)&1)) & table[15];
        *dst++ ^= res;
    }
}

void process_packet_direct_eq16(int log_cte, void *p_dst, void *p_src, int S) {
    memset(p_dst, 0, S);
    process_packet_direct16(log_cte, p_dst, p_src, S);
}


void process_packet_direct8(int log_cte, void *p_dst, void *p_src, int S)
{
    int i;
    int cte = exp_table[log_cte];
    int table[8];
    for (i=0;i<n_field;i++) {
        table[i]=cte;
        cte <<=1;
        if (cte >> n_field)
            cte^=field_mask;
    }

    uint8_t *src = (uint8_t *)p_src;
    uint8_t *dst = (uint8_t *)p_dst;
    for (i=0; i<S; i++)
    {
        int a=*(src++);
        int res;
        res  = (-((a>>0)&1)) & table[0];
        res ^= (-((a>>1)&1)) & table[1];
        res ^= (-((a>>2)&1)) & table[2];
        res ^= (-((a>>3)&1)) & table[3];
        res ^= (-((a>>4)&1)) & table[4];
        res ^= (-((a>>5)&1)) & table[5];
        res ^= (-((a>>6)&1)) & table[6];
        res ^= (-(a>>7)) & table[7];
        *dst++ ^= res;
    }
}

void process_packet_direct_eq8(int log_cte, void *p_dst, void *p_src, int S) {
    memset(p_dst, 0, S);
    process_packet_direct8(log_cte, p_dst, p_src, S);
}


// *******************************************************
// *******************************************************

// these functions need the src packet
// to be in log form

void process_packet_table(int log_cte, void *p_dst, void *p_src, int S)
{
    symbol *table = &exp_table[log_cte];
    int t = table[0];
    table[0]=0;

    int i;
    uint8_t *src = (uint8_t *)p_src;
    uint8_t *dst = (uint8_t *)p_dst;
    for (i=0; i<S; i++)
        *dst++ ^= table[*src++];

    table[0]=t;
}

void process_packet_table_eq(int log_cte, void *p_dst, void *p_src, int S)
{
    symbol *table = &exp_table[log_cte];
    int t = table[0];
    table[0]=0;

    int i;
    uint8_t *src = (uint8_t *)p_src;
    uint8_t *dst = (uint8_t *)p_dst;
    for (i=0; i<S; i++)
        *dst++ = table[*src++];

    table[0]=t;
}

void process_packet_table16(int log_cte, void *p_dst, void *p_src, int S)
{
    symbol *table = &exp_table[log_cte];
    int t = table[0];
    table[0]=0;

    int i;
    uint16_t *src = (uint16_t *)p_src;
    uint16_t *dst = (uint16_t *)p_dst;
    for (i=0; i<S/2; i++)
        *dst++ ^= table[*src++];

    table[0]=t;
}

void process_packet_table_eq16(int log_cte, void *p_dst, void *p_src, int S)
{
    symbol *table = &exp_table[log_cte];
    int t = table[0];
    table[0]=0;

    int i;
    uint16_t *src = (uint16_t *)p_src;
    uint16_t *dst = (uint16_t *)p_dst;
    for (i=0; i<S/2; i++)
        *dst++ = table[*src++];

    table[0]=t;
}

// *******************************************************
// *******************************************************

// used only for special
void (*postprocess)(void *, int)=NULL;

void (*process)(int log_cte, void *dst, void *src, int S)=NULL;
void (*process_eq)(int log_cte, void *dst, void *src, int S)=NULL;

void (*RS_encode)(int N, int K, int S, void *info, void *output)=NULL;
void (*RS_decode)(int N, int K, int S, int *pos, void *received, void *output)=NULL;

// ************************************************
// ************************************************

int n_walsh;
int N_walsh;
symbol *product;
symbol *product_enc;

symbol *log_walsh;
symbol *pos;
symbol *upos;


// Perform a Walsh transform and keep the coeffs mod (modulo)
// The transformation is involutive if N_walsh = N_field.
void walsh_mod(symbol *vect)
{
    int i,j,step;
    step=1;
    while (step<N_walsh) {
        i=0;
        while (i<N_walsh) {
            j = step;
            while (j--)
            {
                int t=vect[i];
                int b=vect[i+step];
                int a=t+b;

                b = t + modulo - b;

                a = (a & modulo) + (a>>n_field);
                b = (b & modulo) + (b>>n_field);

                vect[i]=a;
                vect[i+step]=b;

                i++;
            }
            i+=step;
        }
        step<<=1;
    }
}

void code_init(int nw)
{
    n_walsh = nw;
    if (n_field>31 || n_walsh > n_field) {
        printf("incorrect field parameters\n");
        exit(0);
    }

    N_field = 1<<n_field;
    N_walsh = 1<<n_walsh;
    modulo = N_field-1;

    log_table = (symbol *)malloc(sizeof(symbol)*N_field);
    exp_table = (symbol *)malloc(sizeof(symbol)*2*N_field);

    fill_table(n_field);

    product = (symbol *)malloc(sizeof(symbol)*N_walsh);
    product_enc  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    log_walsh = (symbol *)malloc(sizeof(symbol)*N_walsh);
    pos  = (symbol *)malloc(sizeof(symbol)*N_walsh);
    upos  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    int i;
    for (i=0; i<N_walsh; i++)
        log_walsh[i] = log_table[i] % modulo;
    walsh_mod(log_walsh);

}

void code_clear()
{
    free(log_table);
    free(exp_table);
    free(product);
    free(log_walsh);
    free(pos);
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
    walsh_mod(product);

    // multiplication
    // need long long here if n_field > 16
    // otherwise int is ok.
    for (i=0; i<N_walsh; i++)
        product[i] = ((uint32_t)product[i] * (uint32_t)log_walsh[i]) % modulo;

    // inverse Walsh transform
    // it is not involutive if N_field != N_walsh,
    // so we need to correct it
    walsh_mod(product);
    int shift = n_field - n_walsh;
    for (i=0; i<N_walsh; i++)
        product[i] = ((unsigned int)product[i] << shift) % modulo;
}

// Same but quadratic version
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

// for encoding, we can precompute the product once
void encode_init(int N, int K)
{
    int i;

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[i]=1;

    // compute product
    compute_product();

    // save it in product_enc
    // so it is not overwritten by any decoding
    for (i=0;i<N_walsh; i++)
        product_enc[i] = product[i];
}

// *******************************************************
// *******************************************************

void incremental_encode(int N, int K, int S, void *b_src, void *b_dst)
{
    int i,x;

    for (x=K; x<N; x++)
    {
        void *dst = b_dst + (x-K)*S;
        packet_clear(dst, S);
    }

    for (i=0; i<K; i++)
    for (x=K; x<N; x++)
    {
        void *dst = b_dst + (x-K)*S;
        void *src = b_src + i*S;

        // the second subtraction can also
        // go into quadratic_init
        int t = product_enc[x] - log_table[i ^ x] - product_enc[i];
        if (t<0) t+= modulo;
        if (t<0) t+= modulo;

        process(t, dst, src, S);
    }
}

// find a position p such that pos[p]==0
// remove this position from product and add x to product
int update_product(int x)
{
    int p=0;
    while (p<N_walsh && pos[p]!=0) {
        p++;
    }

    pos[p]=2;

    // update product
    // add new position [x] to set of zero positions
    // remove old position [p] from set of zero positions
    int i;
    for (i=0; i<N_walsh; i++) {
        if (i!=x) product[i] = (product[i] + log_table[i ^ x]) % modulo;
        if (i!=p) product[i] = (product[i] + modulo - log_table[i ^ p]) % modulo;
    }

    return p;
}

// this work only with xor_product
// and only if packet are in order.
void incremental_decode(int N, int K, int S, int *positions, void *b_src, void *b_dst)
{
    int i,j,x;

    for (i=0; i<N_walsh; i++) {
        pos[i]=0;
        product[i] = product_enc[i];
    }

    // loop over received packet.
    for (i=0; i<K; i++)
    {
        // we received a new packet
        // at position [x]
        x = positions[i];

        pos[x]=1;

        // if systematic : just copy.
        if (x<K) {
            memcpy(b_dst + x*S, b_src + i*S , S);
            continue;
        }

        // evaluate current value at positions[i]
        // overwrite b_src with the difference we need.
        for (j=0; j<K; j++) if (pos[j]!=0) {
            int t = product_enc[x] - log_table[j ^ x] - product_enc[j];

            if (t<0) t+= modulo;
            if (t<0) t+= modulo;

            process(t, b_src + i*S, b_dst + j*S, S);
        }

        // update product
        int p=update_product(x);

        // clear new added position
        packet_clear(b_dst + p*S, S);

        // update 'p' values
        for (j=0; j<K; j++) if (pos[j]==2)
        {
            int t = product[j] - log_table[x ^ j] - product[x];
            if (t<0) t+= modulo;
            if (t<0) t+= modulo;

            process(t, b_dst + j*S, b_src + i*S, S);
        }
    }
}

// *******************************************************
// *******************************************************

void quadratic_encode(int N, int K, int S, void *b_src, void *b_dst)
{
    int i,x;

    for (x=K; x<N; x++)
    {
        void *dst = b_dst + (x-K)*S;
        packet_clear(dst, S);

        for (i=0; i<K; i++)
        {
            void *src = b_src + i*S;

            // the second subtraction can also
            // go into quadratic_init
            int t = product_enc[x] - log_table[i ^ x] - product_enc[i];
            if (t<0) t+= modulo;
            if (t<0) t+= modulo;

            process(t, dst, src, S);
        }
    }
}

void quadratic_decode(int N, int K, int S, int *positions, void *b_src, void *b_dst)
{
    int i,x;

    // copy the systematic pieces in place
    for (i=0; i<K; i++)
        if (positions[i]<K) {
            void *dst = b_dst + positions[i]*S;
            void *src = b_src + i*S;
            memcpy(dst, src, S);
        }

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[positions[i]]=1;

    // compute product
    compute_product();

    // decode the other pieces
    for (x=0; x<K; x++) {
        if (pos[x]==0) {
            void *dst = b_dst + x*S;
            packet_clear(dst, S);

            for (i=0; i<K; i++)
            {
                void *src = b_src + i*S;

                // the second subtraction can also
                // go into quadratic_init
                int t = product[x] - log_table[positions[i] ^ x] - product[positions[i]];
                if (t<0) t+= modulo;
                if (t<0) t+= modulo;

                process(t, dst, src, S);
            }
        }
    }
}

void special_encode(int N, int K, int S, void *b_src, void *b_dst)
{
    quadratic_encode(N, K, S, b_src, b_dst);
    int x;
    for (x=K; x<N; x++)
        postprocess(b_dst + (x-K)*S, S);
}

void special_decode(int N, int K, int S, int *positions, void *b_src, void *b_dst)
{
    quadratic_decode(N, K, S, positions, b_src, b_dst);
    int x;
    for (x=0; x<K; x++) {
        if (pos[x]==0) {
            postprocess(b_dst + x*S, S);
        }
    }
}

void karatsuba(void *dest, void *coeff, symbol *inv, int n, int S)
{
    if (n==0) {
        if (inv[0])
            process_eq(log_table[inv[0]], dest, coeff, S);
        else {
            packet_clear(dest,S);
        }
        return;
    }
// TODO : correct this to deals with inv==0
//   if (n==1) {
//        process_eq(log_table[inv[0]], dest, coeff, S);
//        process(log_table[inv[1]], dest, coeff+S, S);
//        memxor(coeff, coeff+S, S);
//        memcpy(dest+S, dest, S);
//        process(log_table[inv[0]^inv[1]], dest+S, coeff, S);
//        memxor(coeff, coeff+S, S);
//        return;
//    }

    int i=0;
    int half=1<<(n-1);
    void *h_dest = dest + half*S;
    void *h_coeff = coeff + half*S;
    symbol  *h_inv = inv + half;

    karatsuba(dest, coeff, inv, n-1, S);
    karatsuba(h_dest, h_coeff, h_inv, n-1, S);

    memxor(dest, h_dest, S*half);
    memxor(h_coeff, coeff, S*half);
    for (i=0; i < half; i++) {
        h_inv[i] ^= inv[i];
    }

    karatsuba(h_dest, h_coeff, h_inv, n-1, S);

    memxor(h_dest, dest, S*half);
    memxor(h_coeff, coeff, S*half);
    for (i=0; i < half; i++) {
        h_inv[i] ^= inv[i];
    }
}

byte *fast_in;
byte *fast_out;
symbol *inverse;

void fast_init(int N_walsh, int S)
{
    fast_in  = (byte *) malloc(sizeof(byte)*N_walsh*S);
    fast_out = (byte *) malloc(sizeof(byte)*N_walsh*S);
    inverse  = (symbol *) malloc(sizeof(symbol)*N_walsh);

    int i;
    for (i=0; i<N_walsh; i++) {
        inverse[i] = exp_table[modulo - log_table[i]];
    }
    inverse[0]=0;
}

void fast_encode(int N, int K, int S, void *b_src, void *b_dst)
{
    int i;

    // clear fast_in
    memset(fast_in, 0, S*N_walsh);

    // compute lagrange coeff, put them in place
    for (i=0; i<K; i++) {
        process_eq(modulo-product_enc[i], fast_in + i*S, b_src + i*S, S);
    }

    // do the convolution with the inverse
//    karatsuba(fast_out, fast_in, inverse, n_walsh, S);
    karatsuba(fast_out+K*S, fast_in, inverse+K, n_walsh-1, S);

    // final multiplication
    for (i=0; i<K; i++) {
        process_eq(product_enc[K+i], b_dst + i*S, fast_out + (K+i)*S, S);
    }
}

void fast_decode(int N, int K, int S, int *positions, void *b_src, void *b_dst)
{
    int i;

    // copy the systematic pieces in place
    for (i=0; i<K; i++)
        if (positions[i]<K) {
            void *dst = b_dst + positions[i]*S;
            void *src = b_src + i*S;
            memcpy(dst, src, S);
        }

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[positions[i]]=1;

    // compute product
    compute_product();

    // clear fast_in
    memset(fast_in, 0, S*N_walsh);

    // compute lagrange coeff, put them in place
    for (i=0; i<K; i++) {
        process_eq(modulo-product[positions[i]], fast_in + positions[i]*S, b_src + i*S, S);
    }

    // do the convolution with inverse
//    karatsuba(fast_out, fast_in, inverse, n_walsh, S);
    karatsuba(fast_out, fast_in, inverse, n_walsh-1, S);
    karatsuba(fast_out+K*S, fast_in+K*S, inverse+K, n_walsh-1, S);
    memxor(fast_out, fast_out + K*S, K*S);

    // final multiplication of unknown pieces
    for (i=0; i<K; i++) {
        if (pos[i]==0) {
            process_eq(product[i], b_dst + i*S, fast_out + i*S, S);
        }
    }
}

// *******************************************************
// *******************************************************

void use_direct()
{
    printf("using direct multiplication\n");
    if (n_field==8) {
        process = &process_packet_direct8;
        process_eq = &process_packet_direct_eq8;
    } else {
        process = &process_packet_direct16;
        process_eq = &process_packet_direct_eq16;
    }
}

void use_xor()
{
    printf("using xor\n");
    process = &process_packet_xor;
    process_eq = &process_packet_xor_eq;
}

void use_xor2()
{
    printf("using other xor\n");
    process = &process_packet_xor2;
    process_eq = &process_packet_xor_eq;
}

void use_table()
{
    printf("using two tables\n");
    if (n_field<=8) {
        process = &process_packet_test;
        process_eq = &process_packet_test_eq;
    } else {
        process = &process_packet_test16;
        process_eq = &process_packet_test16_eq;
    }
    if (n_field==8) {
        printf("using full tabulated multiplication\n");
        process = &process_packet_test8;
        process_eq = &process_packet_test8_eq;
    }
}

// *******************************************************
// *******************************************************

void use_quadratic()
{
    printf("using quadratic algorithm\n");
    RS_encode=quadratic_encode;
    RS_decode=quadratic_decode;
}

void use_incremental()
{
    printf("using incremental algorithm\n");
    RS_encode=incremental_encode;
    RS_decode=incremental_decode;
}

void use_karatsuba()
{
    printf("using karatsuba algorithm\n");
    RS_encode=fast_encode;
    RS_decode=fast_decode;
}

void use_special()
{
    printf("using special quadratic algorithm with 1 table\n");
    if (n_field<=8) {
        process = &process_packet_table;
        process_eq = &process_packet_table_eq;
        postprocess = &packet_log;
    } else {
        process = &process_packet_table16;
        process_eq = &process_packet_table_eq16;
        postprocess = &packet_log16;
    }
    RS_encode=special_encode;
    RS_decode=special_decode;
}
