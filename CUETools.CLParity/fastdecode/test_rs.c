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
#include "stdlib.h"
#include "stdio.h"

int n_field;
int n_walsh;
int N_field;
int N_walsh;
int modulo;

symbol *log_table;
symbol *exp_table;

symbol *product;
symbol *lagrange;
symbol *product_enc;

symbol *log_walsh;
symbol *pos;
symbol *upos;

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
    for (i=0; i<N_field; i++) {
        log_table[i]=0;
        exp_table[i]=0;
    }

    // put the primitive poly in mask
    int temp=0;
    int mask=0;
    int pos=0;
    while (1) {
        if (primitive[pos]==0) {
            mask=0;
            temp=pos;
        }
        mask ^= 1 << primitive[pos];
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

    // usefull since later
    // log_table[1] = modulo
    exp_table[2*modulo]=1;
}

// ************************************************
// ************************************************

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

void code_init(int nf, int nw)
{
    int i;

    n_field = nf;
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

    fill_table();

    product = (symbol *)malloc(sizeof(symbol)*N_walsh);
    lagrange = (symbol *)malloc(sizeof(symbol)*N_walsh);
    product_enc  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    log_walsh = (symbol *)malloc(sizeof(symbol)*N_walsh);
    pos  = (symbol *)malloc(sizeof(symbol)*N_walsh);
    upos  = (symbol *)malloc(sizeof(symbol)*N_walsh);

    for (i=0; i<N_walsh; i++)
        log_walsh[i] = log_table[i];
    walsh_mod(log_walsh);

    // since log_table[0]=0
    // we set log_table[1]=modulo
    // so log_table is a bijection...
    log_table[1]=modulo;
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
        product[i] = ((int)product[i] * (int)log_walsh[i]) % modulo;

    // inverse Walsh transform
    // it is not involutive if N_field != N_walsh,
    // so we need to correct it
    walsh_mod(product);
    int shift = n_field - n_walsh;
    for (i=0; i<N_walsh; i++)
        product[i] = ((int) product[i] << shift) % modulo;
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

// compute one redundancy piece
// this is the heart of the encoding/decoding complexity

// [data] contain the [K] known pieces and the log of the coefficients
// [positions] contain their positions in the codeword
// [x] is the index of the piece we want to compute
// [output] is where we will write the result

// both functions are almost identical
// but it is usefull to have two for profiling

void quadratic_enc(int K, int S, symbol *data, symbol *output, int x)
{
    int i,j;
    int m = product_enc[x];

    // first time we overwrite output
    {
        // the second substraction can also
        // go into quadratic_init
        int t = m - log_table[x] - product_enc[0];
        if (t<0) t+= modulo;
        if (t<0) t+= modulo;

        // we set table[0] = 0
        // because 0 correspond to a null coefficient
        // and not to a logarithm of 0.
        symbol *table = &exp_table[t];
        t = table[0];
        table[0]=0;

        symbol *o = output;

        for (j=0; j<S; j++) {
            *o++ = table[*data++];
        }

        table[0]=t;
    }

    // other time we just xor into the output
    for (i=1; i<K; i++)
    {
        // the second substraction can also
        // go into quadratic_init
        int t = m - log_table[i ^ x] - product_enc[i];
        if (t<0) t+= modulo;
        if (t<0) t+= modulo;

        // we set table[0] = 0
        // because 0 correspond to a null coefficient
        // and not to a logarithm of 0.
        symbol *table = &exp_table[t];
        t = table[0];
        table[0]=0;

        symbol *o = output;

        for (j=0; j<S; j++) {
            *o++ ^= table[*data++];
        }

        table[0]=t;
    }
}

void quadratic_dec(int K, int S, int *positions, symbol *data, symbol *output, int x)
{
    int i,j;
    int m = product[x];

    // first time we erase output
    {
        // the second substraction can also
        // go into quadratic_init
        int t = m - log_table[positions[0] ^ x] - product[positions[0]];
        if (t<0) t+= modulo;
        if (t<0) t+= modulo;

        // we set table[0] = 0
        // because 0 correspond to a null coefficient
        // and not to a logarithm of 0.
        symbol *table = &exp_table[t];
        t = table[0];
        table[0]=0;

        symbol *o = output;

        for (j=0; j<S; j++) {
            *o++ = table[*data++];
        }

        table[0]=t;
    }

    // other time we just xor into the output
    for (i=1; i<K; i++)
    {
        // the second substraction can also
        // go into quadratic_init
        int t = m - log_table[positions[i] ^ x] - product[positions[i]];
        if (t<0) t+= modulo;
        if (t<0) t+= modulo;

        // we set table[0] = 0
        // because 0 correspond to a null coefficient
        // and not to a logarithm of 0.
        symbol *table = &exp_table[t];
        t = table[0];
        table[0]=0;

        symbol *o = output;

        for (j=0; j<S; j++) {
            *o++ ^= table[*data++];
        }

        table[0]=t;
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

// encode
void quadratic_encode(int N, int K, int S, symbol *data, symbol *packets)
{
    int i;

    // precomputation
    // compute the logarithm of the received symbols
    symbol *temp=data;
    for (i=0; i<K*S; i++)
        *temp++ = log_table[*temp];

    // encode
    for (i=0; i<N-K; i++)
        quadratic_enc(K, S, data, &packets[i*S], K+i);
}

// decode
void quadratic_decode(int K, int S, int *positions, symbol *data, symbol *packets)
{
    int i,j;

    // copy the systematic pieces in place
    for (i=0; i<K; i++)
        if (positions[i]<K) {
            symbol *dst = &packets[positions[i]*S];
            symbol *src = &data[i*S];
            for (j=0; j<S; j++)
                *dst++ = *src++;
        }

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[positions[i]]=1;

    // compute product
    compute_product();
//    compute_product_quadratic(K, positions);

    // compute the logarithm of the received symbols
    symbol *temp=data;
    for (i=0; i<K*S; i++)
        *temp++ = log_table[*temp];

    // decode the other pieces
    for (i=0; i<K; i++)
        if (pos[i]==0)
            quadratic_dec(K, S, positions, data, &packets[i*S], i);
}

// *******************************************************
// *******************************************************

int old_pos;

void last_init(int S, symbol *dst, int K, int N)
{
    int i,j;

    for (i=0; i<N_walsh; i++) pos[i]=0;

    for (i=0; i<N_walsh; i++) product[i] = product_enc[i];
    for (i=0; i<N_walsh; i++) lagrange[i] = 0;

    for (i=0; i<N-K; i++) upos[i]=K+i;
    for (i=0; i<N-K; i++) pos[K+i]=2;

    old_pos=-1;
}

void last_step(int S,
    symbol *src, int x,
    symbol *dst, int K, int N)
{
    int i,j;

    pos[x]=1;

    if (x < K)
    {
        // copy packet in place
        // precompute log
        for (j=0; j<S; j++)
        {
            dst[x*S+j] = src[j];
            src[j] = log_table[src[j]];
        }

        // update redundancy packets
        for (i=K; i<N; i++)
        {
            symbol *table= &exp_table[modulo - log_table[x^i]];
            int t = table[0];
            table[0]=0;

            symbol *in = src;
            symbol *out = &dst[i*S];

            if (pos[i]==2)
            {
                for (j=0; j<S; j++) {
                    *out++ = table[*in++];
                }
                pos[i]=3;
            } else {
                for (j=0; j<S; j++) {
                    *out++ ^= table[*in++];
                }
            }

            table[0]=t;
        }
    }
    else
    {
        // precompute log
        for (j=0; j<S; j++) {
            src[j] = log_table[ dst[x*S+j] ^ src[j] ];
        }

        // find next unreceived systematic position
        int p;
        for (i=0; i<N; i++) {
            if (pos[i]==0) {
                p=i;
                break;
            }
        }

        // mark this position to clean it
        pos[p] = 2;
    
        if (old_pos==-1) {
            upos[x-K]=p;
            // we want an 1 at position x.
            for (i=0; i<N-K; i++) {
                if (upos[i]>=K)
                    lagrange[i] = exp_table[modulo - log_table[p ^ upos[i]] + log_table[p ^ x]];
            }
            lagrange[x-K]=p^x;    
        } else {
            // we want a 0 at old_pos (we have a 1 now)
            // we want a 1 at x
            
            // put the 0 
            for (i=0; i<N-K; i++) {
                if (upos[i]>=K)
                lagrange[i] ^= exp_table[modulo - log_table[p ^ upos[i]] + log_table[p ^ old_pos]];
            }
            
            // rescale to get the one
            int c = lagrange[x-K];
            upos[x-K]=p;
            lagrange[x-K]=p ^ old_pos;
            
            for (i=0; i<N-K; i++) {
                if (lagrange[i]!=0)
                    lagrange[i] = exp_table[log_table[lagrange[i]] + modulo - c];
            }
        }
        old_pos = x;

        // update packets
        for (i=0; i<N-K; i++)
        {
            symbol *table= &exp_table[log_table[lagrange[i]]];
            int t = table[0];
            table[0]=0;

            symbol *in = src;
            symbol *out = &dst[upos[i]*S];

            if (pos[upos[i]]==2)
            {
                for (j=0; j<S; j++) {
                    *out++ = table[*in++];
                }
                pos[upos[i]]=3;
            } else {
                for (j=0; j<S; j++) {
                    *out++ ^= table[*in++];
                }
            }

            table[0]=t;
        }
    }
}

// *******************************************************
// *******************************************************

void inc_init(int S, symbol *dst, int K)
{
    int i;

    // init pos
    for (i=0; i<N_walsh; i++)
        pos[i]=0;

    // clear dst
    for (i=0; i<K*S; i++)
        *dst++ = 0;

    for (i=0; i<K; i++) {
        lagrange[i]=product_enc[i];
        product[i]=0;
    }
}

void inc_step(int S,
    symbol *src, int *positions, int R,
    symbol *dst, int K)
{
    int i,j;

    int x = positions[R];
    symbol *p = &src[R*S];

    // fast case :)
    // systematic packet ?
    if (x<K)
    {
        // set pos
        pos[x]=1;

        // copy it in place
        // compute log of coefficient
        for (i=0; i<S; i++) {
            dst[x*S+i] = p[i];
            p[i] = log_table[p[i]];
        }

        for (i=0; i<K; i++) {
           product[i] = (product[i] + log_table[x ^ i]) % modulo;
        }

        // done
        return;
    }

    // add in [p] the value of the previous poly at point [x]
    // m should be equal to product[x] at the end...
    int m=0;
    for (i=0; i<R; i++)
    {
        symbol *table = &exp_table[m];

        if (positions[i]<K) {
            int t = product_enc[x]
                - product_enc[positions[i]]
                - log_table[x ^ positions[i]];
            if (t<0) t+=modulo;
            if (t<0) t+=modulo;
            table = &exp_table[t];
        }

        int t = table[0];
        table[0]=0;

        for (j=0; j<S; j++) {
            p[j] ^= table[*src++];
        }

        table[0]=t;

        m = (m + log_table[positions[i] ^ x]) % modulo;
    }

    // from now on it is almost the same code
    // as in incremental_step

    // precompute log of coefficients - product[x]
    // keep value 0 for 0
    for (j=0; j<S; j++) {
        int t = p[j];
        if (t) {
            t=log_table[t]-m;
            if (t<=0) t+=modulo;
        }
        p[j]=t;
//        p[j]=log_table[p[j]];
    }

    // update pos
    pos[x]=1;

    // update packets
    for (i=0; i<K; i++){
        if (pos[i]==0)
        {
            int t = product[i];// - m;
            if (t<=0) t+= modulo;
            symbol *table= &exp_table[t];

            t = table[0];
            table[0]=0;

            symbol *in = p;

            for (j=0; j<S; j++) {
                *dst++ ^= table[*in++];
            }

            table[0]=t;

        } else {
            dst += S;
        }
    }

    // update product
    // PROBLEM on GF 2^16 ??
    for (i=0; i<K; i++) {
        product[i] = (product[i] + log_table[x ^ i]) % modulo;
    }
}

void inc_step2(int S,
    symbol *src, int *positions, int R,
    symbol *dst, int K)
{
    int i,j;

    int x = positions[R];
    symbol *p = &src[R*S];

    // fast case :)
    // systematic packet ?
    if (x<K)
    {
        // set pos
        pos[x]=1;

        // copy it in place
        // compute log of coefficient
        for (i=0; i<S; i++) {
            dst[x*S+i] = p[i];
            p[i] = log_table[p[i]];
        }

        for (i=0; i<K; i++) {
           product[i] = (product[i] + log_table[x ^ i]) % modulo;
        }

        // done
        return;
    }

    // add in [p] the value of the previous poly at point [x]
    // m should be equal to product[x] at the end...
    int m=0;
    for (i=0; i<R; i++)
    {
        symbol *table = &exp_table[m];

        if (positions[i]<K) {
            int t = product_enc[x]
                - product_enc[positions[i]]
                - log_table[x ^ positions[i]];
            if (t<0) t+=modulo;
            if (t<0) t+=modulo;
            table = &exp_table[t];
        }

        int t = table[0];
        table[0]=0;

        for (j=0; j<S; j++) {
            p[j] ^= table[*src++];
        }

        table[0]=t;

        m = (m + log_table[positions[i] ^ x]) % modulo;
    }

    // from now on it is almost the same code
    // as in incremental_step

    // precompute log of coefficients - product[x]
    // keep value 0 for 0
    for (j=0; j<S; j++) {
        int t = p[j];
        if (t) {
            t=log_table[t]-m;
            if (t<=0) t+=modulo;
        }
        p[j]=t;
//        p[j]=log_table[p[j]];
    }

    // update pos
    pos[x]=1;

    // update packets
    for (i=0; i<K; i++){
        if (pos[i]==0)
        {
            int t = product[i];// - m;
            if (t<=0) t+= modulo;
            symbol *table= &exp_table[t];

            t = table[0];
            table[0]=0;

            symbol *in = p;

            for (j=0; j<S; j++) {
                *dst++ ^= table[*in++];
            }

            table[0]=t;

        } else {
            dst += S;
        }
    }

    // update product
    // PROBLEM on GF 2^16 ??
    for (i=0; i<K; i++) {
        product[i] = (product[i] + log_table[x ^ i]) % modulo;
    }
}

void incremental_init(int S,
    symbol *src, int x,
    symbol *dst, int N)
{
    int i,j;

    // init pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    pos[x]=1;

    // copy packet everywere
    for (i=0; i<N; i++)
    {
        symbol *in = src;
        for (j=0; j<S; j++) {
            *dst++ = *in++;
        }
    }

    // precompute log of coefficients - product[x]
    // Used only in the fast version;
    for (j=0; j<S; j++) {
        *src++ = log_table[*src];
    }

    // update product
    for (i=0; i<N; i++) {
        product[i] = log_table[x ^ i];
    }
}

void incremental_step(int S,
    symbol *src, int x,
    symbol *dst, int N)
{
    int i,j;

    // precompute log of coefficients - product[x]
    // keep value 0 for 0
    int m = product[x];
    for (j=0; j<S; j++) {
        int t = src[j] ^ dst[x*S+j];
        if (t) {
            t=log_table[t]-m;
            if (t<=0) t+=modulo;
        }
        dst[x*S+j] = src[j];
        src[j]=t;
    }

    // update pos
    pos[x]=1;

    // update packets
    for (i=0; i<N; i++){
        if (pos[i]==0)
        {
            symbol *table= &exp_table[product[i]];
            int t = table[0];
            table[0]=0;

            symbol *in = src;

            for (j=0; j<S; j++) {
                *dst++ ^= table[*in++];
            }

            table[0]=t;

        } else {
            dst += S;
        }
    }

    // update product
    // PROBLEM on GF 2^16 ??
    for (i=0; i<N; i++) {
        product[i] = (product[i] + log_table[x ^ i]) % modulo;
    }
}

// R is the index of the current packet
// positions contain their positions.

void incremental_step_fast(int S,
    symbol *src, int *positions, int R,
    symbol *dst, int K)
{
    int i,j;

    int x = positions[R];
    symbol *p = &src[R*S];

    // fast case :)
    if (x<K) {
        incremental_step(S, p, x, dst, K);
        return;
    }

    // add in [p] the value of the previous poly at point [x]
    // m should be equal to product[x] at the end...
    int m=0;
    for (i=0; i<R; i++)
    {
        symbol *table = &exp_table[m];
        int t = table[0];
        table[0]=0;

        for (j=0; j<S; j++) {
            p[j] ^= table[*src++];
        }

        table[0]=t;

        m = (m + log_table[positions[i] ^ x]) % modulo;
    }

    // from now on it is almost the same code
    // as in incremental_step

    // precompute log of coefficients - product[x]
    // keep value 0 for 0
    for (j=0; j<S; j++) {
        int t = p[j];
        if (t) {
            t=log_table[t]-m;
            if (t<=0) t+=modulo;
        }
        p[j]=t;
    }

    // update pos
    pos[x]=1;

    // update packets
    for (i=0; i<K; i++){
        if (pos[i]==0)
        {
            symbol *table= &exp_table[product[i]];

            int t = table[0];
            table[0]=0;

            symbol *in = p;

            for (j=0; j<S; j++) {
                *dst++ ^= table[*in++];
            }

            table[0]=t;

        } else {
            dst += S;
        }
    }

    // update product
    // PROBLEM on GF 2^16 ??
    for (i=0; i<K; i++) {
        product[i] = (product[i] + log_table[x ^ i]) % modulo;
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

void walsh_field(symbol *vect)
{
    int i,j;
    int step=n_field;
    int size=n_field*N_walsh;
    while (step<size) {
        i=0;
        while (i<size) {
            j=step;
            while (j--) {
                symbol temp=vect[i+step];
                vect[i+step] = vect[i] - temp;
                vect[i] +=  temp;
                i++;
            }
            i+=step;
        }
        step<<=1;
    }
}

// extra memory needed for fast evaluation
symbol *coeff;
symbol *inverse;
symbol *codeword;

// initialisation of the memory for fast evaluation
void fast_init()
{
    int i,j;

    codeword = (symbol *) malloc(sizeof(symbol *) * N_walsh);
    inverse = (symbol *) malloc(sizeof(symbol *) * N_walsh * n_field);
    coeff = (symbol *) malloc(sizeof(symbol *) * N_walsh * n_field);

    // fill the inverse table
    for (i=0; i<N_walsh; i++) {
        int t = exp_table[(modulo-log_table[i]) % modulo];
        for (j=0; j<n_field; j++)
            inverse[i*n_field + j] = (t>>j)&1;
    }

    // precompute the Walsh transforms of the inverse
    walsh_field(inverse);
}

void fast_clear()
{
    free(codeword);
    free(coeff);
    free(inverse);
}

// *******************************************************
// *******************************************************

// compute field product of the vector coeff and inverse
// for each [n_field] consecutive int of [inverse] and [coeff]
// multiply them as if they were element of the field in the
// standard basis.
//
// the difference is instead of coefficients in GF_2,
// the coefficients are in Z ...
//
// The other limiting part in the complexity (with walsh_field).
// The multiplication is quadratic in n_field
// and is performed N_walsh time.

void field_product()
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

// Algorithm in roughly
// [n_field * (n_field + 2 n_walsh) * N_walsh] operations and
// [2 n_field * N_walsh]  memory
void fast_code(symbol *res, symbol *product)
{
    int i,j;

    // compute coeff of the lagrange poly
    // Work on GF(2^16), why ??
    for (i=0; i<N_walsh; i++)
    {
        int t=0;
        if (pos[i] && res[i]) {
            t=exp_table[log_table[res[i]] + modulo - product[i]];
        }

        for (j=0; j<n_field; j++) {
            coeff[i*n_field +j] = (t>>j)&1;
        }
    }

    // compute the Walsh transforms of the coefficient
    walsh_field(coeff);

    // multiply with precomputed Walsh transform of the inverse
    field_product();

    // walsh transform again
    walsh_field(coeff);

    // final multiplication by the product
    for (i=0; i<N_walsh; i++) {
        if (pos[i]==0) {
            // reconstruct the field element from
            // its n_field coordinates
            int t=0;
            for (j=0; j<n_field; j++) {
                t ^= ((coeff[i*n_field+j] >> n_walsh) &1 ) << j;
            }

            if (t) {
                res[i] = exp_table[log_table[t] + product[i]];
            }
        }
    }
}

// *******************************************************
// *******************************************************

void fast_encode(int N, int K, int S, symbol *data, symbol *packets)
{
    int i,j;

    // fill pos
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[i]=1;

    // encode codeword by codeword
    for (j=0; j<S; j++)
    {
        // put received symbols in places
        for (i=0; i<K; i++) codeword[i] = data[i*S + j];
        for (i=0; i<N_walsh-K; i++) codeword[K+i] = 0;

        // encode
        fast_code(codeword, product_enc);

        // put decoded symbol in place
        for (i=0; i<N-K; i++) {
            packets[i*S + j] = codeword[K + i];
        }
    }
}

void fast_decode(int K, int S, int *positions, symbol *data, symbol *packets)
{
    int i,j;

    // copy the systematic pieces in place
    for (i=0; i<K; i++) if (positions[i]<K) {
        for (j=0; j<S; j++)
            packets[positions[i]*S + j] = data[i*S +j];
    }

    // precomputation
    for (i=0; i<N_walsh; i++) pos[i]=0;
    for (i=0; i<K; i++) pos[positions[i]]=1;
    compute_product();

    // decode codeword by codeword
    for (j=0; j<S; j++)
    {
        // put received symbols in places
        for (i=0; i<N_walsh; i++) codeword[i]=0;
        for (i=0; i<K; i++) codeword[positions[i]] = data[i*S+j];

        // decode
        fast_code(codeword,product);

        // put decoded symbol in place
        for (i=0; i<K; i++) {
            packets[i*S+j] = codeword[i];
        }
    }
}
