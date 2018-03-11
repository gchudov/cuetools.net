function sum = gfsubstitute(polynomial,value,terms,n,field)

%use: gfsubstitute(polynomial,value,terms,n,field)
%Subtitute i^value in polynomial
%the number of terms in polynomial
%n = n of the decoder

sum = polynomial(1);
for cc = 2:terms
        sum = gfadd(sum,gfmul(polynomial(cc),gfpow(value,cc-1,n),field),field); %Sum all the terms
end
