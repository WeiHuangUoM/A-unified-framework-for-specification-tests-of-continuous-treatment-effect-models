function [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,Ti,Y,weight,U,mhat,gGrad,dlambda,beta)

Nt = size(H,2); %length of t in H(T,t)

X1 = X(:,1:end-1);
X2 = X(:,end); %seperate the continuous and binary covariates

pw = (sqrt(5)-1)/(2*sqrt(5));
r = binornd(1,pw,N,B);
w = (1-sqrt(5))/2+sqrt(5)*r;
%w = normrnd(0,1,[N,B]);

%Estimate E(m(Y,T)|T,X) and E(pi(T,X)m(Y,T)|X)
    EpimX = zeros(N,p);
    EmTX = zeros(N,p);
    for r = 1:p
        fit = srsaddm01(X1,U(:,r),2,10,X2);
        EpimX(:,r) = fit.yhat1; %N x R
        fit = srsaddm01([Ti,X1],mhat(:,r),2,10,X2);
        EmTX(:,r) = fit.yhat1; %N x R
    end
    
    Lhat = zeros(p,p);
    for i = 1:N
        TT = (gGrad(i,:))'*(gGrad(i,:));
        if Y(i) == 0
        term11 =weight(i)*dlambda(i)*TT;
        Lhat(1:p-1,1:p-1) = Lhat(1:p-1,1:p-1)+term11;
        else
        Lhat(1:p-1,1:p-1) = Lhat(1:p-1,1:p-1)-weight(i)*TT;
        Lhat(p,1:p-1)=Lhat(p,1:p-1)+weight(i)*Y(i)*gGrad(i,:);
        Lhat(1:p-1,p)=Lhat(1:p-1,p)+weight(i)*Y(i)*(gGrad(i,:))';
        Lhat(p,p) =Lhat(p,p)- weight(i)*(beta(end)^2+Y(i)^2);
        end
    end
    Lhat = Lhat/N; %p x p
    
    %Estimate L^(-1)*(1/sqrt(N))*sum(S(T_i,X_i)*w_i)
    psi2 = Lhat\(U' - (weight.*EmTX)' + (EpimX')); % p x N
   
    JhatN = zeros(p,Nt);
    phip = zeros(N*Nt,p);
    for l = 1:p
        UHP = U(:,l).*H; %N x Nt
        
        JhatN(l,:) = sum(UHP,1)/sqrt(N); %1 x Nt
        
        piHEmTX = weight.*H.*EmTX(:,l); %NxNt
        
        fit = srsaddm01(X1,UHP(:,1),2,10,X2);
        penwt = fit.penwt(fit.imin1);
        fit = srsaddmV(X1,UHP,2,10,X2,penwt);
        EpimHX = fit.yhat1; %N x Nt
        
        phi = piHEmTX - EpimHX;
        phi = reshape(phi,N*Nt,1); %N*Nt x 1
        
        phip(:,l) = phi; %N*Nt x p
    end
    JhatN = JhatN';
    
    %Estimate E(pi(T,X)mH(T,t)|X)
    JNstar = cell(1,Nt);
    for l = 1:Nt
        Ht = zeros(p,p);
        for i = 1:N
        TT = (gGrad(i,:))'*(gGrad(i,:));
        if Y(i) == 0
        term11 =H(i,l)*weight(i)*dlambda(i)*TT;
        Ht(1:p-1,1:p-1) = Ht(1:p-1,1:p-1)+term11;
        else
        Ht(1:p-1,1:p-1) = Ht(1:p-1,1:p-1)-H(i,l)*weight(i)*TT;
        Ht(p,1:p-1)=Ht(p,1:p-1)+H(i,l)*weight(i)*Y(i)*gGrad(i,:);
        Ht(1:p-1,p)=Ht(1:p-1,p)+H(i,l)*weight(i)*Y(i)*(gGrad(i,:))';
        Ht(p,p) =Ht(p,p)- H(i,l)*weight(i)*(beta(end)^2+Y(i)^2);
        end
        end
        Ht = Ht/N; %p x p
        
        psiN = (Ht*psi2)'; %N x p
        
        UHN = U.*H(:,l); %N x p
        
        phiN = phip((l-1)*N+1:l*N,:);
        
        etaN = UHN  - psiN -phiN;
        
        JNstar{l} = w'*etaN/sqrt(N); %B x p
    end
end