function [gEst,gGrad,thetahat]= CaliEstGeneral(t0,T,weight,Y,p)
% PURPOSE: compute the estimator of the mean effect in the model: E{Y*(t)}
% = g(t,theta), where g(t,theta) =
% theta_0+theta_1*t+...+theta_{p-1}*t^{p-1}.
%--------------------------------------------------------------------------
% USAGE: [gEst,gGrad,thetahat]= CaliEstGeneral(t0,Ti,weight,Y,p)
% where: t0 is a vector of values of treatment where the prediction of the
% potential outcome are wanted (n x 1)
%        T is the vector of the treatment data (N x 1)
%        Y is the vector of the observed outcome (N x 1)
%        weight is the estimated pi_0(T,X) (N x 1)
%        p is the order of the polynomials in the model
%--------------------------------------------------------------------------
% RETURNS: gEst (n x 1) is the vector of linear regression estimator of
% g(t0,thetahat).
%          gGrad (N x p) is the gradient of g in terms of theta valued at
%          T.
%          thetahat (p x 1) is the vector of estimated theata.
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
%    May 27, 2021.
% -------------------------------------------------------------------------

N = length(Y);
T = reshape(T,N,1);
Y = reshape(Y,N,1);
weight = reshape(weight,N,1);

T_poly_mat = repmat(T, 1, p).^repmat(0:(p-1), N, 1);     % N x p
g = @(theta,x_mat) x_mat*theta;


ini = zeros(p,1);
gT = @(theta) g(theta,T_poly_mat);

fun = @(ini) Obj(ini,weight,Y,gT,T_poly_mat);

gs = GlobalSearch('Display','off');
options=optimset('Display','off');
problem = createOptimProblem('fmincon','x0',ini,'objective',fun,'options',options);
thetahat = run(gs,problem);

n = length(t0);
t0_mat = repmat(t0,1,p).^repmat(0:(p-1),n,1);
gEst = g(thetahat,t0_mat);

gGrad = T_poly_mat;

end

function f = Obj(theta,weight,Y,gT,Tv)

gval = gT(theta);
diff = Y - gval;
M_N = mean(weight.*diff.*Tv,1);
f = M_N*M_N';

end

