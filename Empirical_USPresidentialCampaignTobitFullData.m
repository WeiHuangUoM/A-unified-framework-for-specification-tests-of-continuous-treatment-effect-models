%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wei Huang, and Zheng Zhang (2021). 
%     A Unified Framework for Specification Tests of Continuous Treatment
%       Effect Models
%
% Main Matlab codes for empirical applications (continuous treatment).
%
% *****************************************************
% Subroutines used in this main code:
%   
% Data:
%    (1) zip_ads_v4.txt
% *****************************************************
%
% Written by:
%    Wei Huang
%    Lecturer
%    School of Mathematics and Statistics, The University of Melbourne
%
% Last updated:
%    June 18, 2021.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear;
close all;
clc;

tic;                                           % start timing
start_date = now;                              % start date
disp(['Job started: ', datestr(start_date)]);  % display start date
warning('off','all')
%% Step 1: Set-up
B = 500;              % # of bootstrap samples  

p = 6;                 % #of parameters p=6:quintic Tobit model
% fix seed so that we always get the same result
defaultStream = RandStream.getGlobalStream;
defaultStream.reset(3);


%% Step 2: Read Data
load zip_ads_v4.txt
zip = zip_ads_v4(:,1);                         % zip code
ads = zip_ads_v4(:,2);                         % # of political advertisements aired in each zipcode (treatment of interest)
pop = zip_ads_v4(:,3);                         % population
pct_over65 = zip_ads_v4(:,4);                  % percent over age 65
median_income = zip_ads_v4(:,5);               % median household income
pct_black = zip_ads_v4(:,6);                   % percent black
pct_hispanic = zip_ads_v4(:,7);                % percent hispanic
pop_density = zip_ads_v4(:,8);                 % population density
pct_colgrads = zip_ads_v4(:,9)/100;                % percent college graduates
can_commute = zip_ads_v4(:,10);                % binary indicator of whether it is possible to commute to the zip code from a competitive state
contribution = zip_ads_v4(:,11);               % total amount of campaign contribution 
log_pop = log(pop);                            % log of population
log_median_income = log(median_income + 1);    % some elements of median_income are 0
log_pop_density = log(pop_density + 1);        % log of population density
clear median_income pop_density

lambda = FindLamBoxCox(contribution);          % find the best Box-Cox transformation parameter for outcome
Y = BoxCox(contribution,lambda);                     % outcome 
Y = Y-min(Y);

T = log(log(log(ads+1)+1)+2);

X = [log_pop, log_pop_density, log_median_income, pct_over65,...
    pct_hispanic, pct_black, pct_colgrads,can_commute];
clear log_pop log_pop_density log_median_income pct_over65 pct_hispanic pct_black pct_colgrads can_commute                              % sample size

N = length(Y); 
dimX = size(X,2);

Nt=100;
t0 = min(T):(max(T)-min(T))/(Nt-1):max(T);  %For KS statistic

K1 = 2;
K2 = dimX + 1;

%% Step 2.1: Plot raw data (if needed)
figure
hist(contribution)
filename = sprintf('HistContributions.pdf');
xlabel('Contributions','FontSize',20)
ylabel('Histogram','FontSize',20)
set(gca, 'fontsize', 20); set(gcf, 'PaperPosition', [0 0 20 20]);  set(gcf, 'PaperSize', [20 20]);  print(gcf,'-dpdf',filename, '-opengl') 

figure
hist(ads)
filename = sprintf('HistAds.pdf');
xlabel('#ads','FontSize',20)
ylabel('Histogram','FontSize',20)
set(gca, 'fontsize', 20); set(gcf, 'PaperPosition', [0 0 20 20]);  set(gcf, 'PaperSize', [20 20]);  print(gcf,'-dpdf',filename, '-opengl') 


%% Step 2.2: Plot of the transformed data (if needed)
figure
hist(Y)
filename = sprintf('HistTransContributions.pdf');
xlabel('Y','FontSize',20)
ylabel('Histogram','FontSize',20)
set(gca, 'fontsize', 20); set(gcf, 'PaperPosition', [0 0 20 20]);  set(gcf, 'PaperSize', [20 20]);  print(gcf,'-dpdf',filename, '-opengl') 

figure
hist(T)
filename = sprintf('HistTransAds.pdf');
xlabel('log(log(log(#ads+1)+1)+2)','FontSize',20)
ylabel('Histogram','FontSize',20)
set(gca, 'fontsize', 20); set(gcf, 'PaperPosition', [0 0 20 20]);  set(gcf, 'PaperSize', [20 20]);  print(gcf,'-dpdf',filename, '-opengl') 


%% Step 3: Select K and estimate pi_0

% construct polynomials with respect to T and X
umat = [ones(1,N); T';(T.^2)'];               % compute polynomial wrt T (K1 x N) 
vmat = [ones(1,N); X'];    % compute polynomial wrt X (K2 x N)   

umat = umat(1:K1,:);
vmat = vmat(1:K2,:);

% standardize polynomial wrt T
exp_uu = (1/N) * (umat * umat');               % K1 x K1
exp_uu_half = chol(exp_uu, 'lower');
exp_uu_half_inv = exp_uu_half \ eye(K1);
umat_std = exp_uu_half_inv * umat;             % K1 x N
%clear umat exp_uu exp_uu_half exp_uu_half_inv

% standardize polynomial wrt X
exp_vv = (1/N) * (vmat * vmat');               % K2 x K2
exp_vv_half = chol(exp_vv, 'lower');
exp_vv_half_inv = exp_vv_half \ eye(K2);
vmat_std = exp_vv_half_inv * vmat;             % K2 x N
%clear vmat exp_vv exp_vv_half exp_vv_half_inv

weight = get_weight(umat_std',vmat_std',K1,K2);  % estimate pihat 

%% Step 4: Estiamte the parameters and Residual U
[mhat,gGrad,dlambda,beta,gEst]= CaliEstTobit(T,T,Y,T,weight,Y,p);
   
beta(1:end-1) = beta(1:end-1)/beta(end);
beta(end) = 1/beta(end);

U = weight.*mhat; %N x p
%% Step 4.1: Plot the estaimted model (if needed)
tads = 0:1:100;
ttrans = log(log(log(tads+1)+1)+2);
[~,~,~,~,gplot]= CaliEstTobit(ttrans,[],[],T,weight,Y,p);
figure
fig=plot(tads,gplot,'MarkerSize',100,'LineWidth',2);
set(fig,'Color',[0,0,0])
xlabel('#ads','FontSize',15)
ylabel('Y*(t)(BoxCox transformed Contribution)','FontSize',15)
filename = sprintf('PlotEstModel.pdf');
set(gcf, 'PaperPosition', [0 0 16 11]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [16 11]); %Set the paper to have width 5 and height 5.
saveas(gcf, filename, 'pdf') %Save gcfure
   
%% Step 5: Test the model (calculate the p-value)

    %Logit
    Htype = 'Logit';
    
    H = Hfunction(T,T,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueLogit,CMNLogit,CMNstarLogit] = TSCM(JhatN,JNstar,N,B);
    sprintf('pvalueLogit=%f',pvalueLogit)
    
    H = Hfunction(T,t0,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueLogitKS,KSNLogit,KSNstarLogit] = TSKS(JhatN,JNstar,B);
     sprintf('pvalueLogitKS=%f',pvalueLogitKS)
   
    %Fourier
    Htype = 'Sine';
    
    H = Hfunction(T,T,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueSine,CMNSine,CMNstarSine] = TSCM(JhatN,JNstar,N,B);
    sprintf('pvalueSine=%f',pvalueSine)
     
    H = Hfunction(T,t0,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueSineKS,KSNSine,KSNstarSine] = TSKS(JhatN,JNstar,B);
    sprintf('pvalueSineKS=%f',pvalueSineKS)
    
    %Indicator
    Htype = 'Indicator';
    
    H = Hfunction(T,T,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueIndicator,CMNIndicator,CMNstarIndicator] = TSCM(JhatN,JNstar,N,B);
    sprintf('pvalueIndicator=%f',pvalueIndicator)
    
    H = Hfunction(T,t0,Htype);
    [JhatN,JNstar] = JNJNstarTobitUS(N,B,p,H,X,T,Y,weight,U,mhat,gGrad,dlambda,beta);
    [pvalueIndicatorKS,KSNIndicator,KSNstarIndicator] = TSKS(JhatN,JNstar,B);
    sprintf('pvalueIndicatorKS=%f',pvalueIndicatorKS)

 
%% Step 6: Report computational time 
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

%% Step 7: Save Results
filename=sprintf('USPresidentialCampaignTobit_lambda12_%d_weightK1_%d_K2_%d_p_%d.mat',N,K1,K2,p);
save(filename)


