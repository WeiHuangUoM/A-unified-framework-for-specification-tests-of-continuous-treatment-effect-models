function [K1,K2] = CV_K1K2General(umat,vmat,Ti,Y,p)
% PURPOSE: choose K1 and K2 for computing pihat using 10-fold cross
% validation.
%--------------------------------------------------------------------------
% USAGE: [K1,K2] = CV_K1K2General(umat,vmat,Ti,Y,p)
% where: umat, vmat are respectively the polynomials series of T and X.
%        Ti is the treatment variable
%        Y is the observed outcome variable
%        p is the order of polynomial in the model under H0
%--------------------------------------------------------------------------
% RETURNS: K1, K2 are the number (integer) of seive basis used for compute pihat
%--------------------------------------------------------------------------
% SEE ALSO: get_weightCV.m, CaliEstGeneral.m
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

%ndgrid of K1, K2

    Kg = 2:4;

[Kg1,Kg2] = ndgrid(Kg,Kg);
Kg1 = Kg1(:);
Kg2 = Kg2(:);
lK = length(Kg1);

Kfold = 10;
Nvset = N/Kfold;

%T_poly_mat = repmat(Ti, 1, p).^repmat(0:(p-1), N, 1);   

D = zeros(Kfold,lK);
for j = 1:lK
    K1 = Kg1(j);
    K2 = Kg2(j);
    for i = 1:Kfold
        idx = (i-1)*Nvset+1:i*Nvset;
        D_N = mean(diffi(idx,umat,vmat,Y,Ti,K1,K2,p));
        D(i,j) = D_N*D_N;
    end
end
 
G=sum(D,1);
idx = find(G==min(G),1);
K1 = Kg1(idx);
K2 = Kg2(idx);
end

function D = diffi(i,umat,vmat,Y,Ti,K1,K2,p)
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

Ethyes = CaliEstGeneral(Tiv,Ti,weight,Y,p);

D = wv.*(Yv - Ethyes);

end

