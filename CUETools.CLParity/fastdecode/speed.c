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

void generate_message(void *data, int size, int n_field)
{
    if (n_field==8 || n_field==16) {
        int *p = (int *)data;
        size >>=2;
        int i;
        for (i=0; i<size; i++) {
            *p++ = genrand();
        }
    } else {
        unsigned short int *p = (unsigned short int *)data;
        size>>=1;
        int i;
        for (i=0; i<size; i++) {
            *p++ = genrand() & ((1<<n_field)-1);
        }
    }

}

int compare_message(void *p, void *q, int size)
{
    int *pi = p;
    int *qi = q;
    size >>=2;
    int res=0;
    int i;
    for (i=0; i<size; i++) {
        if (pi[i]!=qi[i]) res++;
    }
    return res;
}

// *******************************************************

double get_sec(clock_t diff)
{
    return (double)diff / (double)CLOCKS_PER_SEC;
}

double get_rate(clock_t diff, long long byte)
{
    return (double)(byte)/((double)(1<<20) * get_sec(diff));
}

double get_KB(long long byte)
{
    return (double)byte/(double)(1<<10);
}

// *******************************************************

int main(int argc, char *argv[])
{
    int i,j,b;
    clock_t tick;

    // get parameters
    long long S;
    int n_field,nb_bloc,nb_time;

    // default ones
    S = 1<<10;
    nb_bloc = 1<<14;
    nb_time = 100;

    // help message
    if (argc<=1) {
        printf("usage: %s n_field [S in Byte] [nb bloc] [nb time]\n", argv[0]);
        return 0;
    }

    // read parameters
    n_field = atoi(argv[1]);
    if (argc>2) S = atoi(argv[2])*n_field*4;
    if (argc>3) nb_bloc = atoi(argv[3]);
    if (argc>4) nb_time = atoi(argv[4]);

    // print parameters
    printf("[parameters]\n");
    printf("GF 2^%d\n", n_field);

    printf("packet size = %d Byte\n", S);
    printf("number of packets = %d (%f KB)\n", nb_bloc, get_KB(nb_bloc * S));
    printf("number of time = %d\n", nb_time);
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[initialisation (memory + randomness)]\n");
    tick  = clock();

    // init field
    fill_table(n_field);

    // this is the memory for the packet and their positions
    void *source;
    void *destination;
    int *coeff;

    source =  malloc(S*nb_bloc);
    destination = malloc(S*nb_bloc);
    coeff = malloc(nb_bloc*sizeof(int));

    // init random number generator
    // sgenrand(time(NULL));
    sgenrand(123);

    // Generate the random message
    generate_message(source, S*nb_bloc, n_field);
    for (i=0; i<nb_bloc; i++)
        coeff[i]= genrand() & ((1<<n_field)-1);

    // end of initialisation
    tick = clock() - tick;
    printf("%f s\n", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S*nb_bloc));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[builtin memcpy]\n");
    tick = clock();

    for (j=0; j<nb_time; j++)
    for (i=0; i<nb_bloc; i++)
    {
        __builtin_memcpy(destination+i*S, source+i*S, S);
    }

    tick = clock() - tick;
    printf("%f s\t", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S * nb_bloc * nb_time));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[my memxor]\n");
    tick = clock();

    for (j=0; j<nb_time; j++)
    for (i=0; i<nb_bloc; i++)
    {
        memxor(destination+i*S, source+i*S, S);
    }

    tick = clock() - tick;
    printf("%f s\t", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S * nb_bloc * nb_time));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[packet xor/mult]\n\n");

    long long memory = S * nb_bloc * nb_time;
    double multiplicator;
    
    int test;
    for (test=0; test<5; test++) {
        switch (test) {
            case 0 : use_xor();multiplicator=1;break;
            case 1 : use_xor2();multiplicator=1;break;
            case 2 : use_special();multiplicator=(double)n_field/16.0;break;
            case 3 : use_table();multiplicator=(double)n_field/16.0;break;
            case 4 : use_direct();multiplicator=(double)n_field/16.0;break;
        }
        if (n_field==8) multiplicator=1;

        tick = clock();

        for (j=0; j<nb_time; j++)
        for (i=0; i<nb_bloc; i++)
        {
            process(coeff[i], destination+i*S, source+i*S, S);
        }

        tick = clock() - tick;
        printf("%f s\t", get_sec(tick));
        printf("%f MB/s\n", multiplicator * get_rate(tick, memory));
        printf("\n");
    }

    // end
    return 0;
}

