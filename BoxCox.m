function TransX = BoxCox(x,lambda)
%BoxCox transformation of x
%Written by Wei Huang, Lecturer, University of Melbourne

 lambda1 = lambda(1);
 lambda2 = lambda(2);
 if lambda1 == 0 
    TransX = log(x+lambda2);
 else
    TransX = ((x+lambda2).^lambda1-1)/lambda1;
 end

end