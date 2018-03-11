% Reed-Solomon Errors and Erasures Decoding
% Decoding is based on the Massey-Berlekamp decoder.  Based on the notes of Adina Matache: http://www.ee.ucla.edu/~matache/rsc/slide.html
% This program was written by Jaco Versfeld. For any questions or remarks email Jaco at jaco.versfeld@gmail.com.  
% This code is provided as is and Jaco Versfeld accepts no liability what so ever.
% Feel free to use and modify this code for whatever purpose.

clear
clc


%**************************
%*** RS code Parameters ***
%**************************

% Here we specify the parameters of the (n,k) Reed-Solomon code


m = 4          %Determine the Galois Field, GF(2^m)
n = 2^m - 1    %This is fixed for a Reed-Solomon, the length of the codeword
k = 3          %The number of data symbols, can be anything between 1 to n - 1
h = n-k
t = h/2

%**************************




%*** Generate the Galois Field and Generator polynomial ***

% This step is neccessary for Matlab.  Here we create the Galois Field which is used for
% computations of the Reed-Solomon code


field = gftuple([-1:2^m-2]', m, 2);


%Generator Polynomial:
%Lin + Costello, p.171
%The Generator polynomial is one way of encoding Reed-Solomon codes

%Construct the generator polynomial
c = [1 0]; 
p(1) = c(1);

for i = 1:h-1
    p(1) = gfmul(p(1),1,field);
    p(2) = 0;
    c = gfconv(c,p,field);
end
g = c;

%**************************



%*** RS Encode ***

%Generate Random Data
DATA_IN = randint(1,k,[-1 n-1]);

%RS encoding
parity = RS_ENC4(DATA_IN,n,k,g,field);
RS_CODE = [parity DATA_IN];

%********************************




%*** Channel ***
RECEIVED = RS_CODE


%I introduce the errors manually here, but any channel can be used here, like an AWGN.
% A maximum of 2*t + e <= (n-k) errors and erasures can be corrected, where t is the number of 
% errors and e the number of erasures

%Introduce some errors
RECEIVED(3) = gfadd(RECEIVED(3),randint(1,1,[-1 n-1]),field);
RECEIVED(5) = gfadd(RECEIVED(3),randint(1,1,[-1 n-1]),field);


%Introduce some erasures
erasures = [1 7 9];  %This polynomial contains the positions of the erasures

RECEIVED(1) = -2
RECEIVED(7) = -2
RECEIVED(9) = -2

%****************



%*** Decoding ***

DECODED = RS_E_E_DEC(RECEIVED, erasures,n,k,t,h,g,field);

%****************

if all(DECODED == RS_CODE)
    disp('Decoding Success')
else
    disp('Decoding Failure')
end