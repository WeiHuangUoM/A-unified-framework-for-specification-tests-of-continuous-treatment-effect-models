function [mhat,gGrad,dlambda,thetahat,gEst]= CaliEstTobit(tg,t,y,T,weight,Y,p)
% PURPOSE: compute the estimator of the mean effect in the Tobit model: 
%   Let Y(t) = g(t,theta)+ epislon, where g(t,theta) =
% theta_0+theta_1*t+...+theta_{p-1}*t^{p-1} and epsilon is a mean 0 normal
% random noise. Assume the potential outcome of interest is
% Y*(t) = Y(t) if Y(t)>0 and Y*(t) = 0 if Y(t)<=0.
%--------------------------------------------------------------------------
% USAGE: [mhat,gGrad,dlambda,thetahat,gEst]= CaliEstTobit(tg,t,y,T,weight,Y,p)
% where: tg (n x 1) is a vector of values of treatment where the prediction 
%   of the potential outcomes are wanted (can be empty)
%        t (m x 1) is a vector of values of treatment where the prediction 
%   of the gradient of the log-likelihood function (m) are wanted (can be empty)
%        y (m x 1) is a vector of values of the observed outcome where the
%        prediction of the gradient of the log-likelihood function (m) are 
%        wanted (can be empty).
%        T is the vector of the treatment data (N x 1)
%        Y is the vector of the observed outcomes(N x 1)
%        weight is the estimated pi_0(T,X) (N x 1)
%        p is the number of parameter of the model
%--------------------------------------------------------------------------
% RETURNS:  mhat (N x p) is the gradient of the log-likelihood function in
%           terms of theta valued at Y and T.
%           gGrad (N x p) is the gradient of g in terms of theta valued at
%           T.
%           dlambda (N x 1) is the vector of the first derivative of
%           phi(x)/Phi(x) valued at x = -ghat(T)/sigma.
%           thetahat (p x 1) is the estiamted theta.
%           gEst (n x 1) is the vector of linear regression estimator of
%           g(t0,thetahat).
%--------------------------------------------------------------------------
% SEE ALSO:
% -------------------------------------------------------------------------
% References: 
%    Wei Huang and Zheng Zhang (2021). 
%      A Unified Framework for Specification Tests of Continuous Treatment
%       Effect Models
% -------------------------------------------------------------------------
% Written by:
%    Wei Huang
%    Lecturer
%    School of Mathematics and Statistics, The University of Melbourne
%--------------------------------------------------------------------------
% Last updated:
%    June 18, 2021.
% -------------------------------------------------------------------------

N = length(Y);
T = reshape(T,N,1);
Y = reshape(Y,N,1);
weight = reshape(weight,N,1);

ini = [zeros(p-1,1);1];

fun = @(ini) Epilnf(ini,weight,Y,T,p);
options= optimoptions(@fminunc,'Algorithm','quasi-newton','Display','off');
[thetahat,~] = fminunc(fun,ini,options);

if isempty(t)|| isempty(t)
  mhat = [];
  dlambda = [];
else
[mhat,dlambda] = Predict(thetahat,y,t,p);
end
gGrad = repmat(T,1,p-1).^repmat(0:(p-2),N,1);

if isempty(tg) == 0
    gamma = thetahat(end);
    thetahat(end)=[];

    n = length(tg);
    tg =reshape(tg,n,1);
    t_mat = repmat(tg,1,p-1).^repmat(0:(p-2),n,1);
    gsig = t_mat*thetahat; %N x 1;

    gEst = gsig/gamma;
else
    gEst = [];
end
end

function [glnf,dlambda] = Predict(theta,y0,t0,p)

gamma = theta(end);
theta(end)=[];

n = length(t0);
t0 = reshape(t0,n,1);
t_mat = repmat(t0,1,p-1).^repmat(0:(p-2),n,1);
gsig = t_mat*theta; %n x 1;

Phi = normcdf(-gsig);
phi = normpdf(-gsig);

diff = gamma*y0 - gsig;
gGrad = t_mat; %N x p

term11 = -gGrad.*phi./Phi; %N x (p-1)
term12 = diff.*gGrad; %N x (p-1)

term22 = gamma^(-1) - diff.*y0; %N x 1

term11(y0>0,:) = 0;
term12(y0==0,:) = 0;

term22(y0==0) = 0;

glnf = [term11+term12, term22]; %N x p

dlambda = phi.*gsig./Phi - (phi./Phi).^2; %N x 1

end

function f = Epilnf(theta,weight,Y,Ti,p)

[lnf,~] = Predict(theta,Y,Ti,p);

M_N = mean(weight.*lnf,1);
f = M_N*M_N';

end
