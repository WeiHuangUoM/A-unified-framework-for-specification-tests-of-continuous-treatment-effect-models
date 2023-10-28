function fit = srsaddmV(x,y,degree,nknots,z,penwt)
%   	Modified from David Ruppert's (2000) codes
%       by Wei Huang, 14 June 2021.
%
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
%
%	Calls: powerbasis, quantileknots, addbasis
%
%
%	fit = srsaddmV(x,y,degree,nknots,z,penwt);
%
%
%		INPUT (REQUIRED)
%	x = matrix of independent variables (n by d, where d is the number
%		of independent variables
%	y = set of p-dimensional dependent data (sample size n) (n by p)
%
%		INPUT (OPTIONAL)
%	degree = degree of spline (default is 2)
%	nknots = number of knots for each independent variable (this is
%		d by 1.  If the input value is 1 by 1, then it is expanded
%		to a d by 1 constant vector.  (default is 10 for each
%		independent variable)
%	z = matrix of independent variable that enter linearly (can
%		be empty)
%	penwt = the penalty weight(default is
%		logspace(-8,8,51)')
%
%		OUTPUT
%   	yhat = predicted values
%   	beta = regression coefficients
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

if (nargin < 6 || isempty(penwt) == 1) 
	penwt=logspace(-8,8,51)' ;    %  GRID OF ROUGHNESS PENALTY CONSTANTS
end 

nsubknots = 3 ;

tic ;
itime = 0 ;

[n,d] = size(x) ;

if length(nknots) == 1 
	nknots =nknots*ones(d,1) ;
end 

stdx = std(x) ;
x = x ./  (ones(n,1)*stdx ) ;	% Standardize x-variables so that
				%  common penalties may be appropriate
 
basis1 = addbasis01(x,degree,nknots,z,x,0,1,nsubknots) ; %	For fitting
xm = basis1.xm ; %n x d or n x (d+1)

xx =xm'*xm ; %d x d
xy = xm'*y ; %d x p


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% id is used in the "ridge regression to enforce the
% penalty on the squared sum of jumps in the d-th derivative

id = zeros(1,1+size(z,2)) ;
for j = 1:d 
	id = [ id zeros(1,degree) ones(1,nknots(j)) ] ;
end 

id = diag(id) ;
binv =  xx + penwt*id ; %d x d
beta1 = binv\xy ; %d x p
yhat1 = xm*beta1 ; % n x p

fit=struct('yhat1',yhat1, ...
	'beta1',beta1) ;

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
	
    
