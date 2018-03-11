// ************************************************
// ************************************************

// Fast RS on GF(2^16+1)
// (c) 2009 Frederic Didier.

#include <complex>
using namespace std;

#include "stdio.h"
#include "stdlib.h"
#include "time.h"

typedef unsigned char byte;

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

#define GF ((1<<16)+1)

int *GF_log, *GF_exp, *inv;

int inline reduce(unsigned int a)
{
    a = (a& ((1<<16)-1)) - (a>>16);
    a += (((int)a)>>31) & GF;
    return a;
}

int inline field_mult(unsigned int a, unsigned int b)
{
    if (a==(1<<16)) return -(int)b + (((-(int)b)>>31) & GF);
    return reduce(a*b);
}

// this one work if not both a and b are (1<<16) that is -1.
int inline field_mult_no(unsigned int a, unsigned int b)
{
    return reduce(a*b);
}

int inline field_diff(unsigned int a, unsigned int b)
{
    a -= b;
    return a + ((((int)a)>>31)&GF);
}

int inline field_sum(unsigned int a, unsigned int b)
{
    a -= GF-b;
    return a + ((((int)a)>>31)&GF);
}


void init_field()
{
    GF_log = (int *) malloc(sizeof(int) * GF);
    GF_exp = (int *) malloc(sizeof(int) * GF);
    inv = (int *) malloc(sizeof(int) * GF);

    int p = 1;
    int i;
    for (i=0; i+1<GF; i++) {
        GF_exp[i]=p;
        GF_log[p]=i;
        p = reduce(3*p);
    }
    GF_exp[GF-1]=1;
    GF_log[0]=0;

    for (i=0; i<GF; i++) {
        inv[i]=GF_exp[GF-1-GF_log[i]];
    }
    inv[0]=0;
    inv[1]=1;

    //test
    for (i=0; i<GF; i++) {
        if (field_mult(i, inv[i]) != 1) printf("%d ",i);
        if (GF_exp[GF_log[i]]!=i) printf("%d ", i);
    }
    printf("\n");
}

//*********************************************************************
//*********************************************************************

// return GF_log_2(b);
int get_log(int n)
{
    int i=0;
    while (n>>(i+1))
        i++;
    return i;
}

void reverse(int *vect, int n)
{
    int i,j;

    j=n >> 1;
    for (i=1; i<n; i++)
    {
        if (j > i) {
            int temp=vect[i];
            vect[i]=vect[j];
            vect[j]=temp;
        }

        int m = n >> 1;
        while (m >= 1 && (j & m)) {
            j ^= m;
            m >>= 1;
        }
        j += m;
    }
}

void fft_dit(int *vect, int n)
{
    reverse(vect, n);

    int i,j;

    int step=1;
    int number=n/2;
    int mult=15;
    while (number>0)
    {
        int *p=vect;
        int *q=vect+step;
        for (i=0; i<number; i++) {
            for (j=0; j<step; j++) {
                int a = *p;
                int b = field_mult_no(*q, GF_exp[j<<mult]);
                *(p++) = field_sum(a,b);
                *(q++) = field_diff(a,b);
            }
            p+=step;
            q+=step;
        }
        step<<=1;
        number>>=1;
        mult--;
    }
}

void ifft_dit(int *vect, int n)
{
    reverse(vect, n);

    int i,j;

    int step=1;
    int number=n/2;
    int mult=15;
    while (number>0)
    {
        int *p=vect;
        int *q=vect+step;
        for (i=0; i<number; i++) {
            for (j=2*step; j>step; j--) {
                int a = *p;
                int b = field_mult_no(*q, GF_exp[j<<mult]);
                *(p++) = field_sum(a,b);
                *(q++) = field_diff(a,b);
            }
            p+=step;
            q+=step;
        }
        step<<=1;
        number>>=1;
        mult--;
    }
}


void fft2(int *vect, int n)
{
    int i=n/2;
    while (i--) {
        int a = *vect;
        int b = *(vect+1);
        *(vect++) = field_sum(a,b);
        *(vect++) = field_diff(a,b);
    }
}


// decimation in frequency
void fft_inc(int *vect, int n)
{
    int i,j;

    int number=1;
    int step=n/2;
    int mult = 16 - get_log(n);

    while (step>0)
    {
        int *p=vect;
        int *q=vect+step;
        for (i=0; i<number; i++) {
            for (j=0; j<step; j++) {
                int a = *p;
                int b = *q;
                *(p++) = field_sum(a,b);
    // GF_exp[1<<15] never happen, so we are safe to use mult_no
                *(q++) = field_mult_no(field_diff(a,b), GF_exp[j<<mult]);
            }
            p+=step;
            q+=step;
        }
        step>>=1;
        number<<=1;
        mult++;
    }
}

// decimation in time
void ifft_inc(int *vect, int n)
{
    int i,j;
    int number=n/2;

    int step=1;
    int mult=15;

    int *root=GF_exp + (1<<16);
    while (number>0)
    {
        int *p=vect;
        int *q=vect+step;
        for (i=0; i<number; i++) {
            for (j=0; j<step; j++) {
                int a = *p;
    // GF_exp[1<<15] never happen, so we are safe to use mult_no
                int b = field_mult_no(*q, *(root - (j<<mult)));
                *(p++) = field_sum(a,b);
                *(q++) = field_diff(a,b);
            }
            p+=step;
            q+=step;
        }
        step<<=1;
        number>>=1;
        mult--;
    }
}

void fft_rec(int *vect, int n)
{
    if (n == 1<<11) return fft_inc(vect, n);
/*    if (n==2) {
        int a = vect[0];
        int b = vect[1];
        vect[0] = field_sum(a,b);
        vect[1] = field_diff(a,b);
        return;
    }
*/
    int i;
    int mult = 16 - get_log(n);

    n/=2;
    int *l = vect;
    int *h = vect + n;
    for (i=0; i<n; i++) {
        int a = *l;
        int b = *h;
        *(l++) = field_sum(a,b);
  // GF_exp[1<<15] never happen, so we are safe to use mult_no
        *(h++) = field_mult_no(field_diff(a,b), GF_exp[i<<mult]);
    }

    fft_rec(vect, n);
    fft_rec(vect+n, n);
}

void ifft_rec(int *vect, int n)
{
    if (n <= 1<<11) return ifft_inc(vect, n);
/*    if (n==2) {
        int a = vect[0];
        int b = vect[1];
        vect[0] = field_sum(a,b);
        vect[1] = field_diff(a,b);
        return;
    }
*/
    int i;
    int mult = 16 - get_log(n);
    n/=2;
    ifft_rec(vect, n);
    ifft_rec(vect+n, n);

    int *l = vect;
    int *h = vect + n;
    int *root = GF_exp + (1<<16);
    for (i=0; i<n; i++) {
        int a = *l;
  // GF_exp[1<<15] never happen, so we are safe to use mult_no
        int b = field_mult_no(*h, *(root - (i<<mult)));
        *(l++) = field_sum(a,b);
        *(h++) = field_diff(a,b);
    }
}

void (*fft)(int *, int) = fft_rec;
void (*ifft)(int *, int) = ifft_rec;

//*********************************************************************
//*********************************************************************

void compute_prod_old(int *prod, int *pos, int k, int n)
{
    int x,i;
    for (x=0; x<n; x++) {
        long long t=1;
        for (i=0; i<k; i++) {
            if (x!=pos[i])
                t = field_mult(t, field_diff(x,pos[i]));
        }
        prod[x]=t;
    }
}

void compute_prod(int *prod, int *pos, int k, int n)
{
    int x,i;
    for (x=0; x<n; x++) {
        unsigned int t=0;
        for (i=0; i<k; i++) {
            t += GF_log[field_diff(x,pos[i])];
        }
        prod[x]=GF_exp[t % (1<<16)];
    }
}

// C++ part for FFT, better use a good fft library instead.

const double Pi = acos(-1);

// decimation in frequency
// output is bit reversed
void complex_fft_rec(complex<double>* X, int n)
{
  if (n==1) return;
  for (int i=0; i<n/2; i++) {
    complex<double> a = X[i];
    complex<double> b = X[n/2 + i];
    X[i] = a + b;
    X[n/2 + i] = (a - b) *  polar(1.0, -2*Pi*i/double(n));
  }
  complex_fft_rec(X, n/2);
  complex_fft_rec(X + n/2, n/2);
}

// decimation in time
// input is bit reversed.
void complex_ifft_rec(complex<double>* X, int n)
{
  if (n==1) return;
  complex_ifft_rec(X, n/2);
  complex_ifft_rec(X + n/2, n/2);
  for (int i=0; i<n/2; i++) {
    complex<double> a = X[i];
    complex<double> b = X[n/2 + i] * polar(1.0, 2*Pi*i/double(n));
    X[i] = a + b;
    X[n/2 + i] = (a - b);
  }
}

// The memory can be allocated only once if many call to this
// function are expected. the fft of the log can be precomputed.
void compute_prod_fast(int *prod, int *pos, int k, int n)
{
  const int NN = 2*n;
  complex<double>* R = (complex<double> *)malloc(NN * sizeof(complex<double>));
  complex<double>* L = (complex<double> *)malloc(NN * sizeof(complex<double>));
  for (int i=0; i<NN; ++i) {
    R[i] = 0;
    L[i] = 0;
  }
  for (int i=0; i < k; ++i) {
    R[pos[i]] = 1;
  }
  for (int i=0; i < n; ++i) {
    L[i] = GF_log[i];
    if (i>0) {
       L[NN - i] = GF_log[GF - i];
    }
  }

  // convolution
  complex_fft_rec(R, NN);
  complex_fft_rec(L, NN);
  for (int i=0; i < NN; ++i) {
    R[i] *= L[i];
  }
  complex_ifft_rec(R, NN);

  // now we have the GF_log(prod[i]) in Re(R[i])
  // we take the result mod 2^16 since we are in the multiplicative
  // field of GF(2^16+1)
  for (int x=0; x < n; ++x) {
    prod[x] = GF_exp[((long long) (real(R[x])/double(NN) + 0.5)) % (1<<16)];
  }

  free(R);
  free(L);
}

//*********************************************************************
//*********************************************************************

int *high;
int *low;
int *prod;
int *enc_fft;
int *rev_fft;
int *mid_fft;
int *prod_enc;

void init_code(int n)
{
    low = (int *) malloc(sizeof(int) * n);
    high = (int *) malloc(sizeof(int) * n);
    prod = (int *) malloc(sizeof(int) * n);
    prod_enc = (int *) malloc(sizeof(int) * n);

    enc_fft = (int *) malloc(sizeof(int) * n);
    rev_fft = (int *) malloc(sizeof(int) * n);
    mid_fft = (int *) malloc(sizeof(int) * n);

    int x;
    for (x=0; x<n; x++) {
      enc_fft[x] = inv[x];
      rev_fft[n - x - 1] = inv[GF - x - 1];
      if (x<n/2) {
        mid_fft[x] = inv[x];
      } else {
        mid_fft[x] = inv[GF - n + x];
      }
    }
    fft(enc_fft,n);
    fft(rev_fft,n);
    fft(mid_fft,n);
    // already divide by the correct value, so we
    // will not have to do it after the inverse fft.
    for (x=0; x<n; x++) {
        enc_fft[x] = field_mult(enc_fft[x], inv[n]);
        rev_fft[x] = field_mult(rev_fft[x], inv[n]);
        mid_fft[x] = field_mult(mid_fft[x], inv[n]);
    }
}

//*********************************************************************
//*********************************************************************

void convolution(int *dst, int *src, int n)
{
    int x,y;
    for (x=0; x<n; x++) {
        int t=0;
        for (y=0; y<n; y++) {
            t = (t + field_mult(src[y], inv[(x + GF - y)%GF])) % GF;
        }
        dst[x]= t;
    }
}

//*********************************************************************
//*********************************************************************

void encode(int *dst, int *src, int k, int n)
{
    int x,i;

    // put received packet in place
    // divide by prod[pos[i]]
    for (x=k; x<n; x++) dst[x] = 0;
    for (i=0; i<k; i++) dst[i] = field_mult(src[i], inv[prod_enc[i]]);

    // convolve with inverse function
    // TODO: depending on k we can gain a little because
    // either k<n/2 and the second half of the input is 0
    // either k>=n/2 and we don't need the fist half of the output
    fft(dst, n);
    for (x=0; x<n; x++)
        dst[x] = field_mult(dst[x], enc_fft[x]);
    ifft(dst,n);

    // multiply by prod[x] the parity positions
    for (x=k; x<n; x++) {
        dst[x] = field_mult(dst[x], prod_enc[x]);
    }

    // put systematic symbol in place
    for (i=0; i<k; i++) dst[i]=src[i];
}

// This work like encode except that now, we cannot just
// compute the convolution with one fft since the src position
// are all over the codeword and we need the first k output position.
void decode(int *dst, int *src, int *pos, int k, int n)
{
  int i;

  // put received packet in place and divide by prod[pos[i]]
  // low contains the first half and then is null,
  // high is null and then contains the second half.
  // since we need to take the fft of both, I did some optimisations.

  memset(high, 0, n * sizeof(int));
  memset(low, 0, n * sizeof(int));

  // needed for the first fft pass done here
  int mult = 16 - get_log(n);
  int *root = GF_exp + (1<<16);

  for (i=0; i<k; i++) {
    int v = src[i];
    if (v == 0) continue;
    int x = pos[i];
    v = field_mult(v, inv[prod[x]]);
    // remark that v is non-zero since prod is never 0
    // and taking the inverse will not change this.
    if (x < n/2) {
      low[x] = v;
      low[n/2 + x] = field_mult_no(v, GF_exp[x << mult]);
    } else {
      high[x - n/2] = v;
      high[x] = field_mult_no(GF - v, GF_exp[(x - n/2) << mult]);
    }
  }
  fft(low, n/2);
  fft(low + n/2, n/2);
  fft(high, n/2);
  fft(high + n/2, n/2);

  // compute low part of the convolution
  for (i=0; i<n; i++) {
    dst[i] = field_sum(field_mult(low[i], mid_fft[i]),
                       field_mult(high[i], rev_fft[i]));
  }
  // ifft(dst, n);
  // small optimisation since second part is not needed
  ifft(dst, n/2);
  ifft(dst + n/2, n/2);
  for (i=0; i<n/2; i++) {
    int v = field_mult_no(dst[i + n/2], *(root - (i<<mult)));
    v = field_sum(dst[i], v);
    dst[i] = field_mult(v, prod[i]);
  }

  // compute high part of the convolution
  // only needed if k > n/2
  if (k > n/2) {
    for (i=0; i<n; i++) {
      // we can reuse high as it is no longer needed.
      high[i] = field_sum(field_mult(low[i], enc_fft[i]),
                          field_mult(high[i], mid_fft[i]));
    }
    // small optimisation since first part is not needed
    ifft(high, n/2);
    ifft(high + n/2, n/2);
    for (i=n/2; i<k; i++) {
      int v = field_mult_no(high[i], *(root - ((i-n/2)<<mult)));
      v = field_diff(high[i - n/2], v);
      dst[i] = field_mult(v, prod[i]);
    }
  }

  // replace received positions by the correct symbol
  for (i=0; i<k; i++)
    if (pos[i]<k)
      dst[pos[i]]=src[i];
}

int main(int argc, char *argv[])
{
    int i,j,n;
    clock_t tick;

    // get parameters
    int N, K, S, nb_bloc;

    // help message
    if (argc<=3) {
        printf("usage: %s K N S\n",argv[0]);
        return 0;
    }

    K = atoi(argv[1]);
    N = atoi(argv[2]);
    nb_bloc = atoi(argv[3]);
    S = 2;

    // power of two just greater or equal to N
    int n_walsh=1;
    while ((1<<n_walsh) < N) n_walsh++;
    N = 1<<n_walsh;

    printf("GF(%d)\n", GF);
    printf("K=%d\n",K);
    printf("N=%d\n",N);
    printf("nb_bloc=%d\n", nb_bloc);
    printf("message size = %f KB\n", get_KB(S*K*nb_bloc));

    // ****************************************
    // ****************************************
    printf("[initialisation (memory + randomness)]\n");
    tick  = clock();

    // code init
    init_field();
    init_code(N);

    // memory for the full message
    int *positions = (int *)malloc(sizeof(int)*K*nb_bloc);
    int *message   = (int *)malloc(sizeof(int)*K*nb_bloc);
    int *received  = (int *)malloc(sizeof(int)*K*nb_bloc);


    // init random number generator
     sgenrand(time(NULL));
//    sgenrand(123);
//    sgenrand(321);

    // Generate the random message
    for (i=0; i<K*nb_bloc; i++) {
        message[i] = genrand() % GF;
    }

    // Generate the random positions
//    for (n=0; n<nb_bloc; n++) {
//        generate_positions(N, K, positions + n*K);
//    }
    generate_positions(N, K, positions);

    // memory for encoding/decoding one bloc
    int *codeword = (int *)malloc(sizeof(int)*N);

    // for encoding
    int *kpos=(int *)malloc(sizeof(int)*K);
    for(i=0; i<K; i++) kpos[i] = i;
    compute_prod_fast(prod_enc, kpos, K, N);

    // end of initialisation
    tick = clock() - tick;
    printf("%f s\n", get_sec(tick));
    printf("%f MB/s\n", get_rate(tick, S*K*nb_bloc));
    printf("\n");

    // ****************************************
    // ****************************************
    printf("[encoding message]\n");
    tick = clock();

    int *pos = positions;
    for (n=0; n<nb_bloc; n++)
    {
//        int *pos = positions + n*K;
        int *sys = message + n*K;
        int *rec = received + n*K;

        encode(codeword, sys, K, N);

        // simulate errors...
        for (i=0; i<K; i++) rec[i] = codeword[pos[i]];
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

    compute_prod_fast(prod, pos, K, N);
    for (n=0; n<nb_bloc; n++)
    {
        // current bloc
        int *rec = received + n*K;

        decode(codeword, rec, pos, K, N);

        // put result back into received
        for (i=0; i<K; i++) {
            if (pos[i]<K) syst++;
            rec[i]=codeword[i];
        }
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
    int count=0;
    for (i=0; i<K*nb_bloc; i++) {
        if (message[i]!=received[i]) count++;
    }
    printf("%i\n",count);

    // ****************************************
    // ****************************************

    // end;
    return 0;
}

