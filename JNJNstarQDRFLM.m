function [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U,gGrad,mYg,Partialmg)

w = normrnd(0,1,[N,B]);
%Estimate E(m(Y,g)|T,X) and E(pi(T,X)m(Y,g)*gGrad|X)
    piYgGrad = weight.*mYg.*gGrad; %N x p
    EpiYgGradX = zeros(N,p);
    for r = 1:p
        fit = srsaddm01(X,piYgGrad(:,r));
        EpiYgGradX(:,r) = fit.yhat1; %N x p
    end
    
    fit = srsaddm01([Ti,X],mYg);
    EYTX = fit.yhat1; %N x 1 
    
    Lhat = zeros(p,p);
    for i = 1:N
        Lhat = Lhat + Partialmg(i)*(gGrad(i,:))'*(gGrad(i,:));
    end
    Lhat = Lhat/N; %p x p
    
    %Estimate L^(-1)*(1/sqrt(N))*sum(S(T_i,X_i)*w_i)
    LTw = Lhat\(piYgGrad'*w)/sqrt(N);
    LEXw = Lhat\(EpiYgGradX'*w)/sqrt(N);
    LETXw = Lhat\(((EYTX.*weight.*gGrad)')*w)/sqrt(N);
    LShat = LTw + LEXw - LETXw; %R x B
    
    %Estimate E(pi(T,X)m(Y,g)H(T,t)|X)
    yy = weight.*mYg.*H; %N x Nt
    fit = srsaddm01(X,yy(:,1));
    penwt = fit.penwt(fit.imin1);
    fit = srsaddmV(X,yy,2,10,[],penwt);
    EpiYHX = fit.yhat1;
    
    %Estimate H(t)
    Ht = N^(-1)*(gGrad.*Partialmg)'*H; % p x N
    
    %Estimate H(t)L^(-1)*(1/sqrt(N))*sum(S(T_i,X_i)*w_i)
    HLS = LShat'*Ht; %B x N
    
    %Estimate (1/sqrt(N))*sum(phi(T_i,X_i,t))
    EHXw = w'*EpiYHX/sqrt(N);
    EHTXw = w'*(EYTX.*weight.*H)/sqrt(N);
    phiw = EHXw - EHTXw;
    
    Ustar = w.*U/sqrt(N);
    UstarH = Ustar'*H;
    UH = U'*H;
    JhatN = UH/sqrt(N); %1 x Nt
    
    JNstar = UstarH-HLS + phiw; %B x Nt
    
end