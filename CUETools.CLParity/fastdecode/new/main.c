// ************************************************
// ************************************************

// Sample program to use reed_solomon.c
// (c) 2009 Frederic Didier.

#include "reed_solomon.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

/************************************************/
/** Random number generator -> 32bits          **/
/** Mersenne twister code                      **/
/************************************************/

/* A C-program for MT19937: Integer     version                   */
/*  genrand() generates one pseudorandom unsigned integer (32bit) */
/* which is uniformly distributed among 0 to 2^32-1  for each     */
/* call. sgenrand(seed) set initial values to the working area    */
/* of 624 words. Before genrand(), sgenrand(seed) must be         */
/* called once. (seed is any 32-bit integer except for 0).        */
/*   Coded by Takuji Nishimura, considering the suggestions by    */
/* Topher Cooper and Marc Rieffel in July-Aug. 1997.              */

/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */
/* 02111-1307  USA                                                 */

/* Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.       */
/* Any feedback is very welcome. For any question, comments,       */
/* see http://www.math.keio.ac.jp/matumoto/emt.html or email       */
/* matumoto@math.keio.ac.jp                                        */

/* Period parameters */
#define MT_N 624
#define MT_M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned long mt[MT_N]; /* the table for the state vector  */
static int mti=MT_N+1; /* mti==MT_N+1 means mt[MT_N] is not initialized */

/* initializing the table with a NONZERO seed */
void sgenrand(unsigned long seed)
{
    /* setting initial seeds to mt[MT_N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    for (mti=1; mti<MT_N; mti++)
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;
}

unsigned int genrand()
{
    unsigned int y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= MT_N) { /* generate MT_N words at one time */
        int kk;

        if (mti == MT_N+1)   /* if sgenrand() has not been called, */
            sgenrand(4357); /* a default initial seed is used   */

        for (kk=0;kk<MT_N-MT_M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (;kk<MT_N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[MT_N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return y;
}

double double_genrand() {
    return genrand() * (1.0/4294967295.0);
}

// *******************************************************

// The Art of Computer programming - Knuth
// vol 2 - section 3.4.2 page 137
// Algorithm S (Selection sampling technique)

void generate_positions(int N, int K, int *pos)
{
    int size=N;
    int w=K;
    do {
        if (double_genrand()*size <= w) {
            pos[K-w] = N-size;
            w--;
        }
        size--;
    } while (size);
}

void generate_message(byte *data, int size)
{
    int *p = (int *)data;
    size >>=2;
    int i;
    for (i=0; i<size; i++) {
        *p++ = genrand();
    }
}

int compare_message(byte *p, byte *q, int size)
{
    int *pi = (int *)p;
    int *qi = (int *)q;
    size >>=2;
    int res=0;
    int i;
    for (i=0; i<size; i++) {
        if (pi[i]!=qi[i]) {
            res++;
            if (res<100) {
                printf("%d ",i*4);
            }
        }
    }
    printf("\n");
    return res;
}

// *******************************************************

double get_sec(clock_t diff)
{
    return (double)diff / (double)CLOCKS_PER_SEC;
}

double get_rate(clock_t diff, int byte)
{
    return (double)(byte)/((double)(1<<20) * get_sec(diff));
}

double get_KB(int byte)
{
    return (double)byte/(double)(1<<10);
}

// *******************************************************

int main(int argc, char *argv[])
{
    int i,j,n;
    clock_t tick;

    // get parameters
    int n_field,K,N,S,m_size;

    // default ones
    n_field = 16;
    m_size = 1<<26;
    S = 1<<10;

    // help message
    if (argc<=2) {
        printf("usage: %s K N S [message size in MB]\n",argv[0]);
        return 0;
    }

    // read parameters
    K = atoi(argv[1]);
    N = atoi(argv[2]);
    if (argc>3) S = atoi(argv[3]);
    if (argc>4) m_size = atoi(argv[4]) << 20;

    // number of field elements per packets.
    int nb_elt = (S * 8) / n_field;

    // compute number of blocs
    int nb_bloc = m_size / (K*S);
    if (nb_bloc==0) nb_bloc=1;

    // power of two just greater than N
    int n_walsh=1;
    while ((1<<n_walsh) < N) n_walsh++;

    // print parameters
    printf("[parameters]\n");
    printf("GF 2^%d\n", n_field);
    printf("n = %d (n_walsh = %d)\n", N, n_walsh);
    printf("k = %d\n", K);
    printf("s = %d (bytes per packet)\n", S);

    printf("field elements per packet = %d\n", nb_elt);
    printf("number of bloc = %d\n", nb_bloc);
    printf("bloc    size = %f KB\n", get_KB(K*S));
    printf("message size = %f KB\n", get_KB(K*S*nb_bloc));
    printf("\n");

    // code init
    code_init(n_field, n_walsh);

    // ****************************************
    // ****************************************
    printf("[initialisation (memory + randomness)]\n");
    tick  = clock();

    // memory for the full message
    int  *positions = (int *)malloc(sizeof(int)*K*nb_bloc);
    byte *message  = (byte *)malloc(sizeof(byte)*S*K*nb_bloc);
    byte *received = (byte *)malloc(sizeof(byte)*S*K*nb_bloc);

    // memory for encoding/decoding one bloc
    byte *codeword = (byte *)malloc(sizeof(byte)*S*N);

    // init random number generator
    sgenrand(time(NULL));
//    sgenrand(123);
//    sgenrand(321);

    // Generate the random message
    generate_message(message, S*K*nb_bloc);

    // Generate the random positions
    for (n=0; n<nb_bloc; n++) {
        generate_positions(N, K, positions + n*K);
    }

    // end of initialisation
    tick = clock() - tick;
    printf("%f s\n", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S*K*nb_bloc));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[encoding message]\n");
    tick = clock();

    encode_init(K);

    byte *dst=received;
    for (n=0; n<nb_bloc; n++)
    {
        int  *pos = positions + n*K;
        byte *systematic = message + n*K*S;

        fast_encode(N, K, S/2, (symbol *)systematic, (symbol *)codeword);

        // simulate errors...
        for (i=0; i<K; i++)
        {
            byte *src;
            if (pos[i]<K) src = systematic + pos[i] * S;
            else src = codeword + (pos[i]-K) * S;

            __builtin_memcpy(dst, src, S);
            dst += S;
        }
    }

    tick = clock() - tick;
    printf("%f s\n", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S*K*nb_bloc));
    printf("%f MB/s\n", get_rate(tick, S*N*nb_bloc));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[decoding]\n");
    tick = clock();

    double syst=0;
    for (n=0; n<nb_bloc; n++)
    {
        // current bloc
        int  *pos = positions + n*K;
        byte *rec = received + n*S*K;

        // stat for systematic packet
        for (i=0; i<K; i++) {
            if (pos[i]<K) syst++;
        }

        fast_decode(K, S/2, pos, (symbol *)rec, (symbol *)codeword);

        // put result back into received
        __builtin_memcpy(rec, codeword, S*K);
    }

    tick = clock() - tick;
    printf("%f s\n", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S*K*nb_bloc));
    printf("%f percent of systematic packets received\n", syst / (double)(K*nb_bloc));
    printf("\n");

    // ****************************************
    // ****************************************

    // verify that we recovered the message correctly
    printf("[errors]\n");
    printf("%d\n\n", compare_message(message, received, S*K*nb_bloc));

    // ****************************************
    // ****************************************

    // clear code
    code_clear();

    // end;
    return 0;
}

