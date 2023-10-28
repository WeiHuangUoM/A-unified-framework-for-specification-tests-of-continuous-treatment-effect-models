function [K1,K2] = CV_K1K2Tobit(umat,vmat,Ti,Y,p)
% PURPOSE: choose K1 and K2 for computing pihat using 10-fold cross
% validation for Tobit model.
%--------------------------------------------------------------------------
% USAGE: [K1,K2] = CV_K1K2Tobit(umat,vmat,Ti,Y,p)
% where: umat, vmat are respectively the polynomials series of T and X.
%        Ti is the treatment variable
%        Y is the observed outcome variable
%        p is the number of parameter of the model.
%--------------------------------------------------------------------------
% RETURNS: K1, K2 are the number (integer) of seive basis used for compute pihat
%--------------------------------------------------------------------------
% SEE ALSO: get_weightCV.m, CaliEstTobit.m
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
        D_N = mean(diffi(idx,umat,vmat,Y,Ti,K1,K2,p),1);
        D(i,j) = D_N*D_N';
    end
end
 
G=sum(D,1);
idx = find(G==min(G),1);
K1 = Kg1(idx);
K2 = Kg2(idx);
end

function D = diffi(i,umat,vmat,Y,Ti,K1,K2,p)
Tiv = Ti;
Yv = Y;
umatv = umat;
vmatv = vmat;

Ti(i)=[];
Y(i)=[];
umat(i,:)=[];
vmat(i,:)=[];

weight = get_weightCV(umatv,vmatv,umat,vmat,K1,K2);
wv = weight(i);
weight(i) = [];

[mhatFull,~,~,~] = CaliEstTobit([],Tiv,Yv,Ti,weight,Y,p);
mhat = mhatFull(i,:);

D=wv.*mhat;
end

