function sigma = massey_berlekamp_M2(n,k,t,S,field)

%http://www.ee.ucla.edu/~matache/rsc/node8.html#SECTION00051000000000000000

%Step 2: Initialize variables
kk = 0;



for i = 1:n
    Kappa(1,i) = -Inf;
end
Kappa(1,1) = 0;

%Kappa



LAMBDA = 0;
Tau = [-inf 0];

done = 0;

%Step 3:
while (done ~= 1)
    %disp('K');
    
    kk = kk + 1;
    
    %disp('S(kk)');
    %S(kk)
    
    %disp('LAMBDA')
    %LAMBDA
    
    sum = -Inf;
    for i = 1:LAMBDA
        %Kappa(kk,i+1)
        %S(kk-i)
        sum = gfadd(sum,gfmul(Kappa(kk,i+1),S(kk-i),field),field);
    end
    
    %disp('Delta - sum')
    %sum
    
    delta(kk) = gfadd(S(kk),sum,field);
    
    %disp('delta');
    %delta
    
    %Step 4:
    if (delta(kk) == -Inf)
        for i = 1:n
            Kappa(kk+1,i) = Kappa(kk,i);
        end
    end
    
    
    if (delta(kk) ~= -Inf)
        
        for i = 1:n
            Kappa_i(i) = Kappa(kk-1+1,i);
        end
        
        Kappa_k = gfadd(Kappa_i,gfconv(delta(kk),Tau,field),field);
        
        while length(Kappa_k) < n
            Kappa_k = [Kappa_k -Inf];
        end
        
        for i = 1:length(Kappa_k)
            Kappa(kk+1,i) = Kappa_k(i);
        end
        
        
        %Step 7:
        if (2*LAMBDA < kk)
            LAMBDA = kk - LAMBDA;
            
            for i = 1:n
                Kappa_k(i) = Kappa(kk+1-1,i);
            end
            
            Tau = gfconv(Kappa_k,gfdiv(0,delta(kk),field),field);
        end
    end
    
    %Step 8:
    Tau = gfconv([-Inf 0],Tau,field);
    
    %step 9:
    if kk >= 2*t
        done = 1;
    end
    
    %Kappa
    %LAMBDA
    %Tau
    
    
end  


for i = 1:n
    sigma(i) = Kappa(kk+1,i);
end