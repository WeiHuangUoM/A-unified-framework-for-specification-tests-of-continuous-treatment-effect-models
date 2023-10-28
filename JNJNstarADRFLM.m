function [JhatN,JNstar] = JNJNstarADRFLM(N,B,p,H,X,Ti,Y,weight,U,gGrad,gEst)
% Estimate J_N^0(t) and the bootstraped J_N^*(t)

w = normrnd(0,1,[N,B]);
%Estimate E(Y-g|T,X) and E(pi(T,X)(Y-g)*gGrad|X)
    piYgGrad = weight.*(Y-gEst).*gGrad; %N x p
    EpiYgGradX = zeros(N,p);
    for r = 1:p
        fit = srsaddm01(X,piYgGrad(:,r));
        EpiYgGradX(:,r) = fit.yhat1; %N x p
    end
    
    fit = srsaddm01([Ti,X],Y-gEst);
    EYTX = fit.yhat1; %N x 1 
    
    Lhat = zeros(p,p);
    for i = 1:N
        Lhat = Lhat + (gGrad(i,:))'*(gGrad(i,:));
    end
    Lhat = Lhat/N; %p x p
    
    %Estimate L^(-1)*(1/sqrt(N))*sum(S(T_i,X_i)*w_i)
    LTw = Lhat\(piYgGrad'*w)/sqrt(N);
    LEXw = Lhat\(EpiYgGradX'*w)/sqrt(N);
    LETXw = Lhat\(((EYTX.*weight.*gGrad)')*w)/sqrt(N);
    LShat = LTw + LEXw - LETXw; %p x B
    
    %Estimate E(pi(T,X)(Y-g)H(T,t)|X)
    yy = weight.*(Y-gEst).*H; %N x Nt
    fit = srsaddm01(X,yy(:,1));
    penwt = fit.penwt(fit.imin1);
    fit = srsaddmV(X,yy,2,10,[],penwt);
    EpiYHX = fit.yhat1;
    
    %Estimate H(t)
    Ht = N^(-1)*gGrad'*H; % p x Nt
    
    %Estimate H(t)L^(-1)*(1/sqrt(N))*sum(S(T_i,X_i)*w_i)
    HLS = LShat'*Ht; %B x Nt
    
    %Estimate (1/sqrt(N))*sum(phi(T_i,X_i,t))
    EHXw = w'*EpiYHX/sqrt(N);  %B x Nt
    EHTXw = w'*(EYTX.*weight.*H)/sqrt(N);
    phiw = EHXw - EHTXw;  %B x Nt
    
    Ustar = w.*U/sqrt(N);
    UstarH = Ustar'*H;
    UH = U'*H;
    JhatN = UH/sqrt(N); %1 x Nt
    
    JNstar = UstarH-HLS + phiw; %B x Nt
    
end