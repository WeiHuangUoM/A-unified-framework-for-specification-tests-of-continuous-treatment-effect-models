function fit = srsaddm01(x,y,degree,nknots,z,type,penwt,gcvfact,nsubknots)
% written by David Ruppert (2000)          
%           Smoothed Regression Splines for ADDitive Model with 
%           with a quadratic penalty.  This is equivalent to a ridge
%           regression estimate that shrinks towards a global polynomial
%
%           Computes a d-th degree spline that minimizes the sum of
%           squared residuals plus a roughness penalty equal to a constant times
%           the sum of squared jumps in the d-th derivative
%
%           The constant minimizing gcv on a grid is also found
%
%
%   	Last edited: 5/12/2000
%
%	Calls: powerbasis, quantileknots, addbasis
%
%
%	fit = srsaddm01(x,y,degree,nknots,z,type,penwt,gcvfact,nsubknots);
%
%
%		INPUT (REQUIRED)
%	x = matrix of independent variables (n by d, where d is the number
%		of independent variables
%	y = vector of dependent variable (n by 1)
%
%		INPUT (OPTIONAL)
%	degree = degree of spline (default is 2)
%	nknots = number of knots for each independent variable (this is
%		d by 1.  If the input value is 1 by 1, then it is expanded
%		to a d by 1 constant vector.  (default is 10 for each
%		independent variable)
%	z = matrix of independent variable that enter linearly (can
%		be empty)
%	type = 1 (same global penalty for all components)
%	     = 2 (separate global penalty for all components) (DEFAULT)
%	     = 3 (separate local penalty for all components)
%		All estimates up to and including 'type' are
%		computed, so, for example, type = 2 means that common
%		and separate global penalty estimates are computed.
%
%	penwt = vector of possible values of the penalty weight (actual
%		penalty weight is selected from penwt by GCV) (default is
%		logspace(-8,8,51)')
%	gcvfact = factor in gcv (DEFAULT = 1, which is ordinary gcv)
%	nsubknots = number of subknots for local penalty (has no effect if
%		type < 3) (DEFAULT = 3)
%
%		OUTPUT
%   	yhat = predicted values at gcv lambda
%   	beta = regression coefficients at GCV-lambda.  The coefficients
%		are ordered: intercept, linear terms (in z), polynomial terms
%		for first x, piecewise polynomial terms for first x, etc
%   	mhat = fitted functions at the observed x's (n by d) using 
%		minimum gcv beta.
%	NOTE: yhat, beta, and mhat come in three versions, e.g., beta1,
%		beta2, and beta3.  These are for common global, separate
%		global, and separate local penalties
%          Each fitted function is standardized to sum to 0.
%   	gcv   = gcv at the values of penwt
%   	imin = index of value of penwt that minimizes gcv so that
%          the minimum gcv fitted values and regression coefficients
%          are yhat(:,imin) and beta(:,imin)
%	sig2hat = estimate of sigma^2
%	stderror = standard errors of beta
%	var = covariance matrix of beta
%	t = t-statistics for beta = beta./stderror
%	p = p-values corresponding to t
%
warning('off','all')
if nargin < 3 
	degree = 2 ;    
end 

if nargin < 4 
	nknots = 10 ;
end 

if nargin < 5 
	z = [] ;
end 

if nargin < 6 
	type = 2 ;
end 

if (nargin < 7 || isempty(penwt) == 1) 
	penwt=logspace(-8,8,51)' ;    %  GRID OF ROUGHNESS PENALTY CONSTANTS
end 

if (nargin < 8 || isempty(gcvfact) == 1)
	gcvfact = 1 ;
end 

if (nargin < 9 || isempty(nsubknots) == 1)
	nsubknots = 3 ;
end 

tic ;
itime = 0 ;

[n,d] = size(x) ;

if length(nknots) == 1 
	nknots =nknots*ones(d,1) ;
end 

stdx = std(x) ;
x = x ./  (ones(n,1)*stdx ) ;	% Standardize x-variables so that
				%  common penalties may be appropriate
 
xm=ones(n,1);
if isempty(z) == 0 
	xm = [xm z] ;
end 
basis1 = addbasis01(x,degree,nknots,z,x,0,1,nsubknots) ; %	For fitting
xm = basis1.xm ;
%basis2 = addbasis01(x,degree,nknots,z,x,1) ;	%	For estimating der.
%xmder = basis2.xm ;
knots = basis1.knots ;
subknots = basis1.subknots ;

xx =xm'*xm ;
xy = xm'*y ;

m = max(size(penwt)) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Start computing common global penalty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta1 = zeros(size(xm,2),m) ;
yhat1 = zeros(n,m) ;
asr1 = zeros(m,1) ;
gcv1 = asr1 ;


                % id is used in the "ridge regression to enforce the
                % penalty on the squared sum of jumps in the d-th derivative

id = zeros(1,1+size(z,2)) ;
for j = 1:d 
	id = [ id zeros(1,degree) ones(1,nknots(j)) ] ;
end 

id = diag(id) ;
trsd = zeros(1,m);
	for i=1:m        %  Compute the regression spline for the
                           %  various penalty weights.

	binv =  xx + penwt(i)*id ;
	beta1(:,i) = binv\xy ;
	xxb = binv\xx ;
	trsd(i) = trace(xxb) ;

	yhat1(:,i) = xm*beta1(:,i) ;
	asr1(i) = mean((y-yhat1(:,i)).^2);   %  asr = average squared residual
		%if i==1
		%trsdsd = trace(xxb*xxb) ;
		%sig2hat1= n*asr1(i)/(n-2*trsd(i)+trsdsd) ;
		%end 

		gcv1(i) = asr1(i) / (1-gcvfact*trsd(i)/n)^2;  

	end 
 
imin1 = find(  (gcv1==min(gcv1))  , 1 ) ;
df1 = trsd(imin1) ;
yhat1 = yhat1(:,imin1) ;
beta1 = beta1(:,imin1) ;
alpha1 = penwt(imin1) ;

sig2hat1 = (norm(y-yhat1))^2 / (n - df1) ;
b1 = inv(xx + penwt(imin1)*id) ;
var1 = sig2hat1 * b1*xx*b1 ;
stderror1 = sqrt(diag(var1) + 10*eps) ;
t1 = beta1 ./ stderror1 ;
p1 = 2*(1-tcdf(abs(t1),n-round(df1))) ;


if itime == 1 
	disp('time to compute common global penalty estimate')
	toc
	tic ;
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	END OF SEARCH FOR COMMON GLOBAL PENALTY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	BEGIN SEARCH FOR SEPARATE GLOBAL PENALTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if type > 1 

alpha2 = penwt(imin1)*ones(d,1) ;	%  Vector of alpha's minimizing GCV -
					%  one for each x-variable



niter = 2 ;
for iter = 1:niter 	%	Iterate algorithm

if iter == 1 
	ngrid = 9 ;
	grid = linspace(-4,4,ngrid) ;
	elseif iter == 2  
	ngrid = 5 ;
	grid = linspace(-1,1,ngrid) ;
end 


	for j = 1:d 	%	Iterate over x variables

gcv2 = zeros(ngrid,1) ;
asr2 = gcv2 ;
trsd2 = gcv2 ;
beta2 = zeros(size(xm,2),ngrid) ;
yhat2 = zeros(n,ngrid) ;
		
		for i=1:ngrid        %  Iterate over grid
	
			id2 = zeros(1,1+size(z,2)) ;
				for j2 = 1:d 
					id2 = [ id2 zeros(1,degree) ...
						(alpha2(j2)* ...
					10^(grid(i)*(j2==j))) ...
					*ones(1,nknots(j2)) ] ;
				end 
			
			id2 = diag(id2) ;
			
				binv =  xx + id2;
				beta2(:,i) = binv\xy ;
				xxb = binv\xx ;
				trsd2(i) = trace(xxb) ;
			
				yhat2(:,i) = xm*beta2(:,i) ;
				asr2(i) = mean((y-yhat2(:,i)).^2);  
					if i==1
					trsdsd = trace(xxb*xxb) ;
				sig2hat=n*asr2(i)/(n-2*trsd2(i)+trsdsd) ;
					end 
			
				gcv2(i) = asr2(i) / (1-gcvfact*trsd2(i)/n)^2;  
		
		end 	%	End iteration over grid

		imin2 = find((gcv2==min(gcv2))  , 1 ) ;
		alpha2(j) = alpha2(j) * 10^(grid(imin2)) ;
	
	
		end 	%	End iteration over x variables
	
	      	
	
	
	end 	%	End iteration over algorithm

 %  Compute the regression spline for the
				   % min-gcv penalty
	
	id2 = zeros(1,1+size(z,2)) ;
		for j2 = 1:d 
			id2 = [ id2 zeros(1,degree) ...
				(alpha2(j2))*ones(1,nknots(j2)) ] ;
		end 
	
	id2 = diag(id2) ;
	
	binv2 =  xx + id2;
	xxb2 = binv2\xx ;
	beta2 = binv2\xy ;
	yhat2 = xm*beta2 ;
	asr2 = mean((y-yhat2).^2);   
	df2 = trace(xxb2) ;
	sig2hat2= n*asr2/(n-2*df2+trace(xxb2*xxb2)) ;
	
	b2 = inv(binv2) ;
	var2 = sig2hat2 * b2*xx*b2 ;
	stderror2 = sqrt(diag(var2) + 10*eps) ;
	t2 = beta2 ./ stderror2 ;
	p2 = 2*(1-tcdf(abs(t2),n-round(df2))) ;

if itime == 1 
	disp('additional time to compute separate global penalty estimate')
	toc
	tic ;
end 	%	End of "if itime == 1"

end 	%	End of "if type > 1"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	END OF COMPUTING SEPARATE GLOBAL PENALTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	START COMPUTING SEPARATE LOCAL PENALTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if type > 2 

alpha3 = ones(nsubknots,1)*alpha2' ;	 




niter = 2 ;
for iter = 1:niter 

if iter == 1 
	ngrid = 9 ;
	grid = linspace(-4,4,ngrid) ;
elseif iter == 2 
	ngrid = 5 ;
	grid = linspace(-1,1,ngrid) ;
end 

for j = 1:d 	%	Iterate over x variables
alpha3trial = alpha3 ;

for k = 1:nsubknots 	% iterate over subknots
	

gcv3 = zeros(ngrid,1) ;
asr3 = gcv3 ;
trsd3 = gcv3 ;
beta3 = zeros(size(xm,2),ngrid) ;
yhat3 = zeros(n,ngrid) ;

alpha3trial = alpha3 ;

	for i=1:ngrid        %  Iterate over grid
		alpha3trial(k,j) = alpha3(k,j) * 10^(grid(i)) ;

		id3 = zeros(1,1+size(z,2)) ;
			for j3 = 1:d 

				penalty = interp1(subknots(:,j3), ...
					log10(alpha3trial(:,j3)), ...
					knots(1:nknots(j3),j3),'linear') ;
				id3 = [ id3 zeros(1,degree) 10.^(penalty)' ] ;
			end 
		

		id3 = diag(id3) ;
		
		binv =  xx + id3;

		beta3(:,i) = binv\xy ;

		xxb = binv\xx ;
		trsd3(i) = trace(xxb) ;
	
		yhat3(:,i) = xm*beta3(:,i) ;
		asr3(i) = mean((y-yhat3(:,i)).^2);  
			%if i==1;
			%trsdsd = trace(xxb*xxb) ;
			%sig2hat= n*asr3(i)/(n-2*trsd3(i)+trsdsd) ;
			%end ;
	
		gcv3(i) = asr3(i) / (1-gcvfact*trsd3(i)/n)^2;  
	
	end 	%	End iteration over grid

	imin3 = find((gcv3==min(gcv3))  , 1 ) ;

%[iter j k imin3]
	alpha3(k,j) = alpha3(k,j)*10^(grid(imin3)) ;

end 	%	End iteration over subknots

end 	%	End iteration over x variables

      
	id3 = zeros(1,1+size(z,2)) ;
		for j3 = 1:d 
		
		penalty = interp1(subknots(:,j3), ...
			log10(alpha3(:,j3)), ...
			knots(1:nknots(j3),j3),'linear') ;

		id3 = [ id3 zeros(1,degree) 10.^penalty' ] ;

			
		end 
	
	id3 = diag(id3) ;
	


end 	%	End of iteration of algorithm (over iter)

	binv3 =  xx + id3;
	beta3 = binv3\xy ;
	yhat3 = xm*beta3 ;
	xxb3 = binv3\xx ;
	asr3 = mean((y-yhat3).^2);   
	df3 = trace(xxb3) ;
	sig2hat3= n*asr3/(n-2*df3+trace(xxb3*xxb3)) ;

	b3 = inv(binv3) ;
	var3 = sig2hat3 * b3*xx*b3 ;
	stderror3 = sqrt(diag(var3) + 10*eps) ;
	t3 = beta3 ./ stderror3 ;
	p3 = 2*(1-tcdf(abs(t3),n-round(df3))) ;

if itime == 1 
	disp('additional time to compute separate local penalty estimate')
	toc
end 


end 	%	End of "if type > 3"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	END OF COMPUTING SEPARATE LOCAL PENALTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ngrid = 200 ;
xbar = ones(ngrid,1)*mean(x) ;

if isempty(z) == 0 
	zbar = ones(ngrid,1)*mean(z) ;
else 
	zbar = [] ;
end 


xgrid = zeros(ngrid,d) ;
postvarmhat = xgrid ;
postvarmhatder = xgrid ;

for j=1:d 
	xj = xbar ;
	xderj = 0*xbar ;
	xj(:,j) = linspace(min(x(:,j)),max(x(:,j)),ngrid)' ;
	xderj(:,j) = linspace(min(x(:,j)),max(x(:,j)),ngrid)' ;
	xgrid(:,j) = xj(:,j) ;
	xmj = addbasis(xj,degree,nknots,zbar,x) ;
	xmderj = addbasis(xderj,degree,nknots,zbar,x,1,j) ;	

	mhat1(:,j)=xmj*beta1;
	mhatder1(:,j) = xmderj*beta1  / stdx(j) ;

	if type > 1 
		mhat2(:,j)=xmj*beta2;
		mhatder2(:,j) = xmderj*beta2  / stdx(j) ;
	else 
		beta2 = [] ;
		mhat2 = [] ;
		mhatder2 = [] ;
    end

	if type > 2 
		mhat3(:,j) = xmj*beta3 ;
		mhatder3(:,j) = xmderj*beta3 / stdx(j) ;

	else 
		beta3 = [] ;
		mhat3 = [] ;
		mhatder3 = [] ;
	end 


end 

xgrid = xgrid .* (ones(ngrid,1)*stdx) ;



fit=struct('yhat1',yhat1, ...
	'beta1',beta1, 'beta2', beta2, 'beta3', beta3, ...
	'mhat1',mhat1, 'mhat2', mhat2,'mhat3',mhat3, ...
	'gcv',gcv1, ...
	'imin1',imin1, ...
	'penwt',penwt) ;

function xm = addbasis(x,degree,nknots,z,xforknots,der,jder) ;

%	Outputs derivatives wrt jth x of the spline basis
%
%	USAGE:  xm = addbasis(x,degree,nknots,z,xforknots,der,jder) 
%
%		INPUT (REQUIRED)
%	x = matrix of independent variables (n by d, where d is the number
%		of independent variables
%
%		INPUT (OPTIONAL)
%	degree = degree of spline (default is 2)
%	nknots = number of knots for each independent variable (this is
%		d by 1.  If the input value is 1 by 1, then it is expanded
%		to a d by 1 constant vector.  (default is 10 for each
%		independent variable)
%	z = matrix of independent variable that enter linearly (can
%		be empty --- that is the default)
%	xforknots = x matrix used to create knots (can be the same as
%		x, but can be different --- default is xforknots = x)
%	der = order of the derivative (default = 0)
%	jder = derivative with respect to the jth component of x
%		(default is 1)
%
%
%		OUTPUT
%	xm = derivatives wrt jth x of the additive spline basis

%	Last edit:	1/6/99

[n,d] = size(x) ;

if nargin < 2 ;
	degree = 2 ;
end ;

if nargin < 3 ;
	nknots = 10 ;
end ;

if nargin < 4 ;
	z = [] ;
end ;

if nargin < 5 ;
	xforknots = x ;
end ;

if nargin < 6 ;
	der = 0 ;
end ;

if nargin < 7 ;
	jder = 1 ;
end ;

if der == 0 ;
	xm=ones(n,1);
	else ;
	xm = zeros(n,1) ;
end ;

if isempty(z) == 0 ;
	xm = [xm (der==0)*z] ;
end ;


	for j = 1:d ;
	knots = quantileknots(xforknots(:,j),nknots(j)) ; % Get knots at sample 
							% quantiles
	xmj = powerbasis(x(:,j),degree,knots,der) ;	% GET SPLINE BASIS
	if der > 0 & j ~= jder ;
		xmj = 0*xmj ;
	end ;
	xm = [xm xmj(:,2:size(xmj,2))] ;   
		%  xm is the "design matrix" of the regression spline
	end ;
    

    function basis = addbasis01(x,degree,nknots,z,xforknots,der,jder,nsubknots) ;
%
%	Copied from addbasis.m
%
%	Outputs derivatives wrt jth x of the spline basis
%
%	USAGE:  xm = addbasis01(x,degree,nknots,z,xforknots,der,jder,nsubknots) 
%
%		INPUT (REQUIRED)
%	x = matrix of independent variables (n by d, where d is the number
%		of independent variables
%
%		INPUT (OPTIONAL)
%	degree = degree of spline (default is 2)
%	nknots = number of knots for each independent variable (this is
%		d by 1.  If the input value is 1 by 1, then it is expanded
%		to a d by 1 constant vector.  (default is 10 for each
%		independent variable)
%	z = matrix of independent variable that enter linearly (can
%		be empty --- that is the default)
%	xforknots = x matrix used to create knots (can be the same as
%		x, but can be different --- default is xforknots = x)
%	der = order of the derivative (default = 0)
%	jder = derivative with respect to the jth component of x
%		(default is 1)
%	nsubknots = number of subknots for each x variable (scalar)
%
%
%		OUTPUT
%	xm = derivatives wrt jth x of the additive spline basis

%	Last edit:	7/12/99

[n,d] = size(x) ;

if (nargin < 2 | isempty(degree) == 1 ) ;
	degree = 2 ;
end ;

if ( nargin < 3 | isempty(nknots) == 1 ) ;
	nknots = 10 ;
end ;

if nargin < 4  ;
	z = [] ;
end ;

if ( nargin < 5 | isempty(xforknots) == 1 );
	xforknots = x ;
end ;

if ( nargin < 6 | isempty(der) == 1 ) ;
	der = 0 ;
end ;

if ( nargin < 7 | isempty(jder) == 1 );
	jder = 1 ;
end ;

if ( nargin < 8 | isempty(nsubknots) == 1 ) ;
	nsubknots = 0 ;
end ;

if der == 0 ;
	xm=ones(n,1);
	else ;
	xm = zeros(n,1) ;
end ;

if isempty(z) == 0 ;
	xm = [xm (der==0)*z] ;
end ;

if length(nknots) == 1 ;
	nknots = ones(d,1)*nknots ;
end ;
maxnknots = max(nknots) ;
knots = zeros(maxnknots,d) ;


if nsubknots > 0 ;
	subknots = zeros(nsubknots,d) ;
else ;
	subknots = [] ;
end ;


for j = 1:d ;
	xunique = unique(xforknots(:,j)) ;

	knots(1:nknots(j),j) = quantileknots(xunique, ...
		nknots(j)) ;

	if nsubknots > 0 ;
		xunique2 = xunique(  (xunique>knots(1,j)) & ...
			(xunique<knots(nknots(j),j)) ) ;
	
		subknots2 = quantileknots(xunique2,nsubknots-2) ;

		subknots(:,j) = [knots(1,j); subknots2; knots(nknots(j),j)] ;
	end ;


	xmj = powerbasis(x(:,j),degree,knots(1:nknots(j),j), ...
			der) ;	% GET SPLINE BASIS
	if der > 0 & j ~= jder ; 
		xmj = 0*xmj ;
	end ;
	xm = [xm xmj(:,2:size(xmj,2))] ;   
		%  xm is the "design matrix" of the regression spline
end ;

basis = struct('xm',xm,'knots',knots,'subknots',subknots) ;

function xm = powerbasis(x,degree,knots,der) ;
%
%	Returns the power basis functions of a spline of given degree
%	USAGE: xm = powerbasis(x,degree,knots) 
%
%	Last edit: 	1/4/99
%

if nargin < 4 ;
	der = 0 ;
end ;

if der > degree ;
disp('********************************************************') ;
disp('********************************************************') ;
disp('WARNING:  der > degree --- xm not returned by powerbasis') ;
disp('********************************************************') ;
disp('********************************************************') ;
return ;
end ;

n=size(x,1) ;
nknots = length(knots);

if der == 0 ;
	xm=ones(n,1);
	else ;
	xm = zeros(n,1) ;
end ;

	for i=1:degree ;
		if i < der ;
			xm = [xm zeros(n,1)] ;
			else ;
			xm = [xm prod((i-der+1):i) *  x.^(i-der)] ;
		end ;
	end ;
	
	if nknots > 0 ;
		for i=1:(nknots) ;
		xm = [xm prod((degree-der+1):degree) * ...
			 (x-knots(i)).^(degree-der).*(x > knots(i))] ;
		end ;
	end ;
    
function knots = quantileknots(x,nknots,boundstab) 

%	Create knots at sample quantiles.  If boundstab == 1, then nknot+2
%	knots are created and the first and last are deleted.  This
%	mitigates the extra variability of regression spline estimates near
%	the boundaries.
%	
%		INPUT (required)
%	x = independent variable.  (The knots are at sample quantiles of x.)
%	nknots = number of knots
%
%		INPUT (optional)
%	boundstab = parameter for boundary stability (DEFAULT is 0)
%
%	USAGE: knots = quantileknots(x,nknots,boundstab) ;
%
%
%	Last edit: 9/16/20
%
if nargin < 3 
	boundstab = 0 ;
end 
x = unique(x) ;
n = length(x) ;
xsort = sort(x) ;   


loc = n*(1:nknots+2*boundstab)' ./ (nknots+1+2*boundstab) ;
knots=xsort(round(loc)) ;
knots=knots(1 + boundstab : nknots + boundstab) ;  
		%  REMOVE KNOTS NEAR BOUNDARIRES FOR
				%  STABILITY (= LOW VARIABILITY)
	
    
