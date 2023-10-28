function weight = get_weightCV(umatv,vmatv,umat,vmat,K1,K2)
% PURPOSE: estimate the pihat in 10-fold CV for quantile regression
%--------------------------------------------------------------------------
% USAGE: weight = get_weightCV(umatv,vmatv,umat,vmat,K1,K2)
% where: umat, vmat are respectively the polynomials series of (validation/prediction) t (n x K1) and x (n x K2).
%        umat, vmat are respectively the polynomials series of (training) T (N x K1) and X (N x K2).
%        K1, K2 are the number of seive basis
%--------------------------------------------------------------------------
% RETURNS: weight = vector of pihat (n x 1)
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
%    November 26, 2020.
% -------------------------------------------------------------------------

ini = zeros(K1,K2);

N=size(umat,1);

umat = umat(:,1:K1); %(N x K1) matrix of polynomials of Ti

vmat = vmat(:,1:K2); %(N x K2) matrix of polynomials of X

%orthogonomalize basis matrix
exp_uu = (1/N) * (umat' * umat);               % K1 x K1
exp_uu_half = chol(exp_uu, 'lower');
exp_uu_half_inv = exp_uu_half \ eye(K1);
umat_std = umat * exp_uu_half_inv'; 
    
exp_vv = (1/N) * (vmat' * vmat);               % K2 x K2
exp_vv_half = chol(exp_vv, 'lower');
exp_vv_half_inv = exp_vv_half \ eye(K2);
vmat_std = vmat * exp_vv_half_inv';


[LamHat,~] = optlam(ini,umat_std,vmat_std,N); %Output the (K1 x K2) matrix of the
                                      %estimator of Lambda 

umatv = umatv(:,1:K1); %(n x K1) matrix of polynomials of t
vmatv = vmatv(:,1:K2); %(n x K2) matrix of polynomials of x
%orthogonomalize basis matrix
umatv_std = umatv * exp_uu_half_inv'; 
vmatv_std = vmatv * exp_vv_half_inv';

u = umatv_std*LamHat*vmatv_std'; % (n x n) matrix of pihat(t,x)

weight = drho(u); %obtain the weight pi_i
weight = diag(weight);
end



function [lamhat,fval] = optlam(ini,umat,vmat,N)

options= optimoptions(@fminunc,'Algorithm','trust-region','StepTolerance',1e-10,'SpecifyObjectiveGradient',true,'Display','off');
fun = @(lam)Obj(lam,umat,vmat,N);
[lamhat,fval] = fminunc(fun,ini,options);


end


function [f,g] = Obj(Lam,umat,vmat,N)

meanumat = mean(umat,1);
meanvmat = mean(vmat,1);

u = umat*Lam.*vmat;
v = sum(u,2);
Y = rho(v);

term1 = mean(Y);
term2 =meanumat*Lam*meanvmat';

f = -term1 + term2; %Function G(Lambda)

if nargout > 1 % supply gradient
    a = drho(v);
    c = vmat.*a;
    term1 = umat'*c/N;
    term2 = meanumat'*meanvmat;
    g = -term1 + term2; %Gradient of G(Lambda)
end

end

function r = rho(x)
r = -exp(-x-1);
end

function dr = drho(x)
dr = exp(-x-1);
end