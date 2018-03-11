function a = gfpow(value,p,n)
%a = gfpow(alpha,p,n)
%alpha^value, (alpha^value)^p  and n = 2^m- in Galois field
a = mod(value*p,n);