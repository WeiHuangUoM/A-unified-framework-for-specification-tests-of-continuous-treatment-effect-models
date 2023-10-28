function weight = get_weight(umat,vmat,K1,K2)
% PURPOSE: estimate the pihat
%--------------------------------------------------------------------------
% USAGE: weight = get_weight(umat,vmat,K1,K2)
% where: umat, vmat are respectively the polynomials series of T and X.
%        K1, K2 are the number of seive basis
%--------------------------------------------------------------------------
% RETURNS: weight = vector of pihat (N x 1)
%--------------------------------------------------------------------------
% SEE ALSO: 
% -------------------------------------------------------------------------
% References: 
%    Wei Huang and Zheng Zhang (2021). 
%        A Unified Framework for Specification Tests of Continuous Treatment
%       Effect Models
% -------------------------------------------------------------------------
% Written by:
%    Wei Huang
%    Lecturer
%    School of Mathematics and Statistics, The University of Melbourne
%--------------------------------------------------------------------------
% Last updated:
%    November 10, 2020.
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
weight = zeros(N,1);
for i = 1:N
     uvec_std = umat_std(i,:);            % 1 x K1 
     vvec_std = vmat_std(i,:);            % 1 x K2
     quad_form_u_Lambda_v = uvec_std * LamHat * vvec_std';         % this matrix approach is faster than the double-sum approach
     weight(i) = drho(quad_form_u_Lambda_v);    % derivative of rho
end

end



function [lamhat,fval] = optlam(ini,umat,vmat,N)

options= optimoptions('fminunc','Algorithm','trust-region','StepTolerance',1e-10,'SpecifyObjectiveGradient',true,'Display','off');
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





