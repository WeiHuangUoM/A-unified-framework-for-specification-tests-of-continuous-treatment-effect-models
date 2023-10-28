function [Y,X,Ti,Uy] = DGP1L(N,Seed)

rng(Seed);

X = rand(N,1);
Ut = normrnd(0,1,[N,1]);
Uy = normrnd(0,1,[N,1]);

Ti = 1+0.2*X + Ut;

Y = 1 + 0.1*Ti.^3 + X +Uy;

end