function [Y,X,Ti,Uy] = DGP0L(N,Seed)

rng(Seed);

X = rand(N,1);
Ut = normrnd(0,1,[N,1]);
%Ut = rand(N,1);
Uy = normrnd(0,1,[N,1]);

Ti = 1+0.2*X + Ut;

Y = 1 + Ti + X +Uy;

end