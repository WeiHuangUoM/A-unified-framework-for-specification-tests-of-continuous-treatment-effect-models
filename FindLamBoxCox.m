function lambda = FindLamBoxCox(Y)

 ini = [0,1];
 lb = [-1,1e-4];
 ub = [1,10];
 %ini = 0;
 options= optimoptions('fmincon','Display','off');
 fun = @(lam)Obj(lam,Y);
 lambda = fmincon(fun,ini,[],[],[],[],lb,ub,[],options);

end

function f = Obj(lambda,Y)
    
    TransY = BoxCox(Y,lambda);
    mu = mean(TransY);
    sigma = std(TransY);
    normTransY = (TransY - mu)/sigma;
    
    qY = quantile(normTransY,(0.1:0.0001:0.9));
    
    qN = norminv((0.1:0.0001:0.9));

    f = corrcoef(qY,qN);
    f = -(f(1,2)).^2;

end