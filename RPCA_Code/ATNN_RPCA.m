function [X,E_hat,iter] = ATNN_RPCA(Ten, opts)
%(r, lambda, tol, maxIter)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

% addpath PROPACK;

[m,n,p] = size(Ten);
if ~exist('opts','var')
    opts=[]; 
end
if isfield(opts,'r')
    r = opts.r; 
else
    r = 10; 
end
if isfield(opts,'maxIter')
    maxIter = opts.maxIter; 
else
    maxIter = 200; 
end
if isfield(opts,'rho')
    rho = opts.rho; 
else
    rho = 1.05; 
end
if isfield(opts,'tol')
    tol = opts.tol; 
else
    tol = 1e-5; 
end
if isfield(opts,'lambda')
    lambda = opts.lambda; 
else
    lambda = 1/sqrt(max(m,n)); 
end

% initialize
D = reshape(Ten,[m*n,p]);
Y = D;
[u_ini, s_ini,v_ini]=svd(D,'econ');
U_hat = u_ini(:,1:r)*s_ini(1:r,1:r);
V_hat = v_ini(:,1:r);
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = U_hat*V_hat';
E_hat = zeros( m*n, p);
mu = 1/norm_two; % this one can be tuned
mu_bar = mu * 1e7;
rho = 1.05;          % this one can be tuned
d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
sv = 10;
while ~converged       
    iter = iter + 1;
    temp_T = D - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);
    % update U_hat
    tmp = D - E_hat + (1/mu)*Y;
    tmpU = tmp*V_hat;
    tnnL = 0;
    for i=1:r
        [U,S,V]=svd(reshape(tmpU(:,i),[m,n]),'econ');
        diagS = diag(S);
        svp = length(find(diagS > 1/mu));
        tnnL = tnnL + sum(diagS(1:svp));
        tmpU_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';   
        U_hat(:,i)=tmpU_hat(:);
    end
    obj = tnnL+lambda*norm(E_hat(:),1);
    [U,~,V] = svd(tmp'*U_hat, 'econ');
    V_hat = U*V';
    total_svd = total_svd + 1;
    A_hat = U_hat *V_hat';
    
    Z = D - A_hat - E_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
    if iter ==1 || mod( total_svd, 40) == 0
        disp([   '#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' 0bj ' num2str(obj)...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
X = reshape(A_hat,[m,n,p]);
E_hat = reshape(E_hat,[m,n,p]);

