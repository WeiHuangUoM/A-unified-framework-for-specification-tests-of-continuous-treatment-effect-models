function [gEst,gGrad,theta]= CaliEstQuantile(t0,T,Y,weight,p,tau)
% PURPOSE: compute the estimator of the tau-th quantile effect in the model:
% E{tau - I[Y*(t)<= g(t,theta)]}=0 ,where g(t,theta) =theta_0+theta_1*t+...
% +theta_{p-1}*t^{p-1} and I(.) is the indicator function.
%--------------------------------------------------------------------------
% USAGE: [gEst,gGrad,theta]= CaliEstQuantile(t0,Ti,Y,weight,p,tau)
% where: t0 is a vector of values of treatment where the prediction of
% outcome are wanted (n x 1)
%        T is the vector of the treatment data (N x 1)
%        Y is the vector of the observed outcome data (N x 1)
%        weight are the estimated pi_0(T,X) (N x 1)
%        p is the order of polynomial in the model
%        tau is the quantile level in (0,1)
%--------------------------------------------------------------------------
% RETURNS: gEst (n x 1) is the vector of linear regression estimator of
% g(t0,thetahat).
%          gGrad (N x p) is the gradient of g in terms of theta valued at
%          T.
%          theta (p x 1) is the estimated theta.
%--------------------------------------------------------------------------
% SEE ALSO:
% -------------------------------------------------------------------------
% References: 
%    Wei Huang and Zheng Zhang (2021). 
%       A Unified Framework for Specification Tests of Continuous Treatment
%       Effect Models
% -------------------------------------------------------------------------
% Written by:
%    Wei Huang
%    Lecturer
%    School of Mathematics and Statistics, The University of Melbourne
%--------------------------------------------------------------------------
% Last updated:
%    May 30, 2021.
% -------------------------------------------------------------------------

N = length(Y);
T = reshape(T,N,1);
Y = reshape(Y,N,1);

T_poly_mat = repmat(T, 1, p).^repmat(0:(p-1), N, 1);     % N x p
g = @(theta,x_mat) x_mat*theta;
gGrad = T_poly_mat;

ini = zeros(p,1);
gT = @(par) g(par,T_poly_mat);

fun = @(par) M(par,Y,weight,gT,tau,gGrad);
options = optimoptions('fsolve','Display','off');
ini = fsolve(fun,ini,options);
lb = ini - abs(quantile(weight.*Y,tau)/(mean(T)));
ub = ini + abs(quantile(weight.*Y,tau)/(mean(T)));

fun = @(par) Obj(par,Y,weight,gT,tau,gGrad);

gs = GlobalSearch('Display','off');
options=optimset('Display','off');
problem = createOptimProblem('fmincon','x0',ini,'objective',fun,'lb',lb,'ub',ub,'options',options);
theta = run(gs,problem);

n = length(t0);
t0_mat = repmat(t0,1,p).^repmat(0:(p-1),n,1);
gEst = g(theta,t0_mat);


end

function Mn = M(theta,Y,weight,gT,tau,gGrad)
N = length(Y);
h = 2.5*N^(-1/2);
gval = gT(theta);
diff = Y - gval;
indi = I(-diff/h);
Mn = mean(weight.*(tau-indi).*gGrad,1);
end

function f = Obj(theta,Y,weight,gT,tau,gGrad)
N=length(Y);
h = 2.5*N^(-1/2);
gval = gT(theta);
diff = Y - gval;
indi = I(-diff/h);

M = weight.*(tau-indi).*gGrad;
M_N = mean(M,1);
f = M_N*M_N';

end

function sI = I(u)

I1 = u<=1 & u>=-1;
I2 = u>1;

fu = 0.5+105/64*(u-5/3*u.^3+7/5*u.^5-3/7*u.^7);

sI = I1.*fu +I2;
end



