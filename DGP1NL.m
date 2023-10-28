function [Y,X,Ti,Uy] = DGP1NL(N,Seed)

rng(Seed);

X = rand(N,1);
Ut = normrnd(0,1,[N,1]);
Uy = normrnd(0,1,[N,1]);

Ti = 0.1*X.^2 + Ut;

Y = 0.2*Ti.^3 + X.^2 +Uy;

end