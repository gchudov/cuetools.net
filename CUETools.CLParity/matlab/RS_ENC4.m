function R = RS_ENC4(code,n,k,g,field)

%R = RS_ENC4(code,n,k,g,field)

%R is the parity block that is included

% m  is the number of bits of each symbol
% n = 2^m-1 => the number of symbols transmitted
% k = the number of code symbols that is going to be codes to a n symbol message
% t = the number of errors that can be found + corrected

%Tripple-error-correcting Reed-Solomon code with symbols from GF(2^4)
% Lin & Costello p.175 and article: Reed_Solomon Codes by Joel Sylvester

%generator polynomial

%field = gftuple([-1:2^m-2]', m, 2);

%p = 2; m = 4;
%primpoly = [0 0 -Inf -Inf 0];
%field = gftuple([-1:p^m-2]',primpoly,p);


%Lin + Costello, p.171


%Encoder (Article)
%shift codeword by X^(n-k)
for ii = 1:n-k
    shiftpol(ii) = -Inf;
end
shiftpol(n-k+1) = 0;
shiftcode = gfconv(code,shiftpol,field);


%divide shifted codeword by g(x)
[Q, R] = GFDECONV(shiftcode, g, field);

while length(R) < n-k
    R = [R -inf];
end

%for ii = 1:n-k
%    CON(ii) = -Inf;
%    if length(R) >= ii
%        CON(ii) = gfadd(R(ii),CON(ii),field);
%    end
%end

%%concatenate the parity to the data
%message = [CON code];
