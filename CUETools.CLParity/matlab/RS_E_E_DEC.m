%h = n - k = npar
%t = h / 2

function DECODED = RS_E_E_DEC(received, erasures,n,k,t,h,g,field);

%Check for decoding failures
%Previous decoder RS_E_E_DEC4

%****************
%*** Decoding ***
%****************

%syndrome calculation
S = [];
%Subtitute alpha^i in received polynomial - Lin + Costello p.152 eq. 6.13
for ii = 1:2*t
    S(ii)= -Inf;
    for cc = 1:n
        S(ii) = gfadd(S(ii),gfmul(received(cc),gfpow(ii,cc-1,n),field),field); %Sum all the terms
    end
end
%S

%Test if syndrome  = 0, if syndrome equals 0, assume that no errors occured
for i = 1:2*t
    test_pol(i) = -Inf;
end

if all (S == test_pol)
    
    message = received;
    
    for i = 1:n
        if message(i) < 0
            message(i) = -Inf;
        end
    end
    
else
    
    
    %Compute the erasure locator polynomial:
    erasures_pos = erasures - 1;
    num_erasures = length(erasures);
    
    %Compute the erasure-locator polynomial
    erasure_loc_pol = 0;
    for i = 1:length(erasures_pos)
        erasure_loc_pol = gfconv(erasure_loc_pol, [0 erasures_pos(i)],field);
    end
    
    %Compute modified syndrome polynomial:
    S_pol = [-inf S];
    dividend = gfconv(erasure_loc_pol,gfadd(0,S_pol,field),field);
    dividend = gfadd(dividend,0,field);
    
    
    divisor = [];
    for i = 1:2*t+2
        divisor(i) = -Inf;
    end
    divisor(2*t+2) = 0;
    
    [q,mod_syn] = gfdeconv(dividend,divisor,field);
    
    while length(mod_syn) < h+1
        mod_syn = [mod_syn -Inf];
    end
    
    S_M = [];
    for i = 1:h - num_erasures
        S_M(i) = mod_syn(i + num_erasures + 1);
    end
    
    flag = 0;
    if isempty(S_M) == 1
        flag = 0;
    else
        for i = 1:length(S_M)
            if (S_M(i) ~= -Inf)
                flag = 1;     %Other errors occured in conjunction with erasures
            end
        end
    end
    
    
    
    %Find error-location polynomial sigma (Berlekamp's iterative algorithm - 
    %sigma = [0 7 4 6]
    if (flag == 1)
        %sigma = M_B2(n,k,length(S_M) - 1,S_M,field);
        
        
        num_iter = t - num_erasures/2;
        
        
        sigma = massey_berlekamp_M3(n,k,num_iter,S_M,field);
        
        %Chien search
        %step 3 from Lin + Costello p.175
        %the error locating polynomial have a maximum of t entries
        error_loc = [];
        kk = 0;
    
        for ii = 0:n-1
            error_r = -Inf;
            for cc = 1:length(sigma)
                error_r = gfadd(error_r,gfmul(sigma(cc),gfpow(ii,cc-1,n),field),field); %Sum all the terms
            end
            if error_r == -Inf
                kk = kk + 1;
                error_loc(kk) = ii;
            end          
        end
        
        
        % Test if the roots are distinct
        % Form a test polynomial by multiplying the roots of error_loc with each other
        % Divide the error_loc pol by test pol
        % if the degree of the quotient exceeds a constant, then the roots are 
        % not distinct
        
        test_pol = 0;
        for ii = 1:length(error_loc)
            test_pol = gfconv(test_pol,[error_loc(ii) 0],field);
        end
        
        %test_pol
        %error_loc
        %sigma
        
        [QQ,RR] = gfdeconv(sigma,test_pol,field);
        if length(QQ) > 1
            DECODED = received;
            return 
        end
            
    
    
    
    
        comp_error_locs = [];    
        %Take reciprocals of elements in error_loc - error location numbers
        for ii = 1:length(error_loc)
            comp_error_locs(ii) = gfdiv(0,error_loc(ii),field);
        end
        %error_loc_p %places where errors occur
    else
        sigma = 0;
        comp_error_locs = [];
    end







    %Calculate error magnitudes - Forney algorithm?
    %Step 4. Lin and Costello - This program uses another algorithm from: 
    %            drake.ee.washington.edu/~adina/rsc/slide/node9.html
    %            http://www.ee.ucla.edu/~matache/rsc/slide.html
    %Compute the error magnitude polynomial:
    %1.  Form the function [1 + S(x)]

    SS(1) = 0;
    for ii = 1: 2*t
        SS(ii+1) = S(ii);
    end

    %SS


    %2. form the product of SS and the KEY Equation
    %OMEGA = gfconv(SS,sigma,field);
    OMEGA = gfconv(sigma,gfadd(0,mod_syn,field),field);



    %3. OMEGA = (SS * sigma)mod(x^(2t+1))
    %3.1. Form a function := x^(2t+1)
    for ii = 1: (2*t)
        DIV(ii)= -Inf;
    end
    DIV(2*t+1) = 0;


    %3.2.  OMEGA = (SS * sigma)mod(x^(2t+1))
    [DUMMY, OMEGA] = gfdeconv(OMEGA,DIV,field);
    %OMEGA

    %4. Differentiate the key equation with respect to x
    %sigma_diff = gfdiff(sigma);
    tsi = gfconv(sigma,erasure_loc_pol,field);
    tsi_diff = gfdiff(tsi);
    
    e_e_places = [erasures_pos comp_error_locs];

    %Calculate the error magnitudes
    %Substitute the inverse into sigma_diff
    for ii = 1:length(e_e_places)
        %error_loc_p(ii)
        ERR_DEN = gfsubstitute(tsi_diff,gfdiv(0,e_e_places(ii),field),length(tsi_diff),n,field);
        ERR_NUM = gfsubstitute(OMEGA,gfdiv(0,e_e_places(ii),field),length(OMEGA),n,field);
        ERR_NUM = gfmul(ERR_NUM,e_e_places(ii),field);
        ERR(ii) = gfmul(ERR_NUM,gfdiv(0,ERR_DEN,field),field);
    end

    %error_loc_p
    %ERR

    %Determine introduced error
    for ii = 1:n
        ERR_p(ii) = -Inf;
    end

    %Error -  t must be substituted by amount of errors 
    for ii = 1:length(e_e_places)
        pp = e_e_places(ii);
        ERR_p(pp+1) = ERR(ii);
    end

    %ERR_p

    message = gfadd(received,ERR_p,field);
    
end

DECODED = message;