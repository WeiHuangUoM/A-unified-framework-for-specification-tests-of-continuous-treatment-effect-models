function [K1,K2] = CV_K1K2Quantile(umat,vmat,Ti,Y,p,tau)
% PURPOSE: choose K1 and K2 for computing pihat using 10-fold cross
% validation.
%--------------------------------------------------------------------------
% USAGE: [K1,K2] = CV_K1K2Quantile(umat,vmat,Ti,Y,p,tau)
% where: umat, vmat are respectively the polynomials series of T and X.
%        Ti is the treatment variable
%        Y is the observed outcome variable
%        p is the order of polynomial in the model under H0
%        tau is the quantile level in (0,1)
%--------------------------------------------------------------------------
% RETURNS: K1, K2 are the number (integer) of seive basis used for compute pihat
%--------------------------------------------------------------------------
% SEE ALSO: get_weightCV.m, CaliEstQuantile.m
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

%ndgrid of K1, K2
Kg = 2:4;

[Kg1,Kg2] = ndgrid(Kg,Kg);
Kg1 = Kg1(:);
Kg2 = Kg2(:);
lK = length(Kg1);

Kfold = 10;
Nvset = N/Kfold;

D = zeros(Kfold,lK);
for j = 1:lK
    K1 = Kg1(j);
    K2 = Kg2(j);
    for i = 1:Kfold
        idx = (i-1)*Nvset+1:i*Nvset;
        D_N = mean(diffi(idx,umat,vmat,Y,Ti,K1,K2,p,tau));
        D(i,j) = D_N*D_N;
    end
end
 
G=sum(D,1);
idx = find(G==min(G),1);
K1 = Kg1(idx);
K2 = Kg2(idx);
end

function D = diffi(i,umat,vmat,Y,Ti,K1,K2,p,tau)
Tiv = Ti(i);
Yv = Y(i);
umatv = umat;
vmatv = vmat;

Ti(i)=[];
Y(i)=[];
umat(i,:)=[];
vmat(i,:)=[];

weight = get_weightCV(umatv,vmatv,umat,vmat,K1,K2);

wv = weight(i);
weight(i) = [];


N = length(Y);
Ti = reshape(Ti,N,1);
Y = reshape(Y,N,1);

T_poly_mat = repmat(Ti, 1, p).^repmat(0:(p-1), N, 1);     % N x p
g = @(theta,x_mat) x_mat*theta;
gGrad = T_poly_mat;

ini = zeros(p,1);
gT = @(par) g(par,T_poly_mat);

fun = @(par) M(par,Y,weight,gT,tau,gGrad);
options = optimoptions('fsolve','Display','off');
theta = fsolve(fun,ini,options);
n = length(Tiv);
t0_mat = repmat(Tiv,1,p).^repmat(0:(p-1),n,1);
Ethyes = g(theta,t0_mat);

h = 2.5*n^(-1/2);
diff = Yv - Ethyes;
indi = I(-diff/h);
D = wv.*(tau - indi);

end

function Mn = M(theta,Y,weight,gT,tau,gGrad)
N=length(Y);
h = 2.5*N^(-1/2);
gval = gT(theta);
diff = Y - gval;
indi = I(-diff/h);
Mn = mean(weight.*(tau-indi).*gGrad,1);
end

function sI = I(u)

I1 = u<=1 & u>=-1;
I2 = u>1;

fu = 0.5+105/64*(u-5/3*u.^3+7/5*u.^5-3/7*u.^7);

sI = I1.*fu +I2;
end
