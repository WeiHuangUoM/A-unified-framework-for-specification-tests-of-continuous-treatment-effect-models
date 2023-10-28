%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wei Huang and Zheng Zhang (2021).
%  A Unified Framework for Specification Tests of Continuous Treatment
%       Effect Models
%
% Main Matlab codes for Monte Carlo simulations on rejection probabilities 
%   with a single covariate (Quantile function).
%
% *****************************************************
% Subroutines directly used in this main code:
%    (1) CaliEstQuantile
%    (2) get_weight.m
%    (3) CV_K1K2Quantile.m
%    (4) Hfunction.m
%    (5) Data generate process files: DGP0L.m, DGP0NL.m, DGP1L.m, DGP1NL.m
%    (6) JNJNstarADRFLM.m TSCM.m TSKS.m polyFun.m
% *****************************************************
%
% Written by:
%    Wei Huang
%    Lecturer
%    School of Mathematics and Statistics, The University of Melbourne
%
% Last updated:
%    June 11, 2021.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear
close all
%clc

tic;                                           % start timing
start_date = now;                              % start date
disp(['Job started: ', datestr(start_date)]);  % display start date

warning('off','all')
%% Step 1: Set-up
J = 1000;               % # of Monte Carlo samples
B = 500;               % # of bootstrap samples  
p = 2;                 % order of linear model (p = 2: g(T)=theta_0 + theat_1*T

tau = 0.5;              % quantile level (tau = 0.5: median)
DGP = "0L";            % 1. DGP0-L: T = 1 + 0.2 * X + xi
                        %            Y = 1 + X + T + epsilon
                        % 2. DGP0-NL: T = 0.1 * X^2 + xi
                        %             Y = X^2 + T + epsilon
                        % 3. DGP1-L: T = 1 + 0.2 * X + xi
                        %            Y = 1 + X + 0.1*T^3 + epsilon
                        % 4. DGP1-NL: T = 0.1 * X^2 + xi
                        %             Y = X^2 + 0.2*T^3 + epsilon
N = 500;                % sample size: 100, 200, 500
%% Step 2: Run Monte Carlo Simulations
parfor j = 1:J
   
    %j
%% step 2.1: Generate and save data    
    if strcmp(DGP,'0L')
        [Y,X,Ti,Uy] = DGP0L(N,j);
    elseif strcmp(DGP,'0NL')
        [Y,X,Ti,Uy] = DGP0NL(N,j);
    elseif strcmp(DGP,'1L')
        [Y,X,Ti,Uy] = DGP1L(N,j);
    elseif strcmp(DGP,'1NL')
        [Y,X,Ti,Uy] = DGP1NL(N,j);
    end
    
    YDG(:,j)=Y;
    XDG(:,j)=X;
    TDG(:,j)=Ti;
    
    % construct polynomial with respect to T
    umat = polyFUN(Ti); %N*K_1 matrix of the u_{K_1}(S_i)'s
    % construct polynomial with respect to X
    vmat = polyFUN(X); %N*K_2 matrix of the v_{K_2}(X_i)'s

    %select #seive basis K1 and K2
    [K1(j),K2(j)] = CV_K1K2Quantile(umat,vmat,Ti,Y,p,tau);
    
    weight = get_weight(umat,vmat,K1(j),K2(j));
    
%% step 2.3: Estiamte the parameters and Residual U

    [gEst,gGrad,thetahat(:,j)]= CaliEstQuantile(Ti,Ti,Y,weight,p,tau);

    indi = Y<=gEst;
    mYg = tau - indi;
    U(:,j) = weight.*mYg;
    hn = 0.9686*std(Y)*N^(-1/7);
    Kh = normpdf((Y-gEst')/hn)/hn; %N x n
    Partialmg = zeros(N,N);
    piKh = weight.*Kh;
    for i = 1:N
       fit = srsaddm01(Ti,piKh(:,i));
       Partialmg(:,i) = fit.yhat1;
    end
    Partialmg = diag(Partialmg);
    
%% step 2.4: Test the model (calculate the p-value)

    %Logit
    Htype = 'Logit';
    
    H = Hfunction(Ti,Ti,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueLogit(j),CMNLogit(j),CMNstarLogit(:,j)] = TSCM(JhatN,JNstar,N,B);
    
    t0 = min(Ti):(max(Ti)-min(Ti))/(100-1):max(Ti);
    H = Hfunction(Ti,t0,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueLogitKS(j),KSNLogit(j),KSNstarLogit(:,j)] = TSKS(JhatN,JNstar,B);
    
    %Fourier
    Htype = 'Sine';
    
    H = Hfunction(Ti,Ti,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueSine(j),CMNSine(j),CMNstarSine(:,j)] = TSCM(JhatN,JNstar,N,B);
    
    t0 = min(Ti):(max(Ti)-min(Ti))/(100-1):max(Ti);
    H = Hfunction(Ti,t0,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueSineKS(j),KSNSine(j),KSNstarSine(:,j)] = TSKS(JhatN,JNstar,B);
    
     %Logit
    Htype = 'Indicator';
    
    H = Hfunction(Ti,Ti,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueIndicator(j),CMNIndicator(j),CMNstarIndicator(:,j)] = TSCM(JhatN,JNstar,N,B);
    
    t0 = min(Ti):(max(Ti)-min(Ti))/(100-1):max(Ti);
    H = Hfunction(Ti,t0,Htype);
    [JhatN,JNstar] = JNJNstarQDRFLM(N,B,p,H,X,Ti,weight,U(:,j),gGrad,mYg,Partialmg);
    [pvalueIndicatorKS(j),KSNIndicator(j),KSNstarIndicator(:,j)] = TSKS(JhatN,JNstar,B);
    
   
end

Rej001Logit = pvalueLogit < 0.01;
Rej005Logit = pvalueLogit < 0.05;
Rej01Logit = pvalueLogit < 0.1;


Rej001Sine = pvalueSine < 0.01;
Rej005Sine = pvalueSine < 0.05;
Rej01Sine = pvalueSine < 0.1;

Rej001Indicator = pvalueIndicator < 0.01;
Rej005Indicator = pvalueIndicator < 0.05;
Rej01Indicator = pvalueIndicator < 0.1;

Rej001LogitKS = pvalueLogitKS < 0.01;
Rej005LogitKS = pvalueLogitKS < 0.05;
Rej01LogitKS = pvalueLogitKS < 0.1;

Rej001SineKS = pvalueSineKS < 0.01;
Rej005SineKS = pvalueSineKS < 0.05;
Rej01SineKS = pvalueSineKS < 0.1;

Rej001IndicatorKS = pvalueIndicatorKS < 0.01;
Rej005IndicatorKS = pvalueIndicatorKS < 0.05;
Rej01IndicatorKS = pvalueIndicatorKS < 0.1;

filename=sprintf('DGP%sQuantile-N%d-tau%.2f-%s.mat',DGP,N,tau,date);
save(filename)
%% Step 3: Report computational time 
time = toc;        % finish timing
end_date = now;    % end date
disp('*****************************************');
disp(['Job started: ', datestr(start_date)]);
disp(['Job finished: ', datestr(end_date)]);
disp(['Computational time: ', num2str(time), ' seconds.']);
disp(['Computational time: ', num2str(time / 60), ' minutes.']);
disp(['Computational time: ', num2str(time / 3600), ' hours.']);
disp('*****************************************');
disp(' ');

%load gong.mat;
%sound(y)
