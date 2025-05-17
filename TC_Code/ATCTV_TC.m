function [X,X_best,S,iter] = ATCTV_TC(Ten, mask, r,tol, maxIter)

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
if nargin < 3
    r = 10;
end

if nargin < 6
    tol = 1e-4;
elseif tol == -1
    tol = 1e-4;
end

if nargin < 5
    maxIter = 300;
elseif maxIter == -1
    maxIter = 300;
end

D = reshape(Ten,[m*n,p]);
normD = norm(D(:));
%[u,s,v]=svd(D,'econ');
U_mat = randn(m*n,r);%u(:,1:r)*s(1:r,1:r);
V_mat = randn(p,r);%v(:,1:r);
%U_mat = randn(m*n,r);%u(:,1:r)*s(1:r,1:r);
%V_mat = randn(p,r);%v(:,1:r);
X = U_mat*V_mat';
sizeU = [m,n,r];
G1    = reshape(diff_x(U_mat,sizeU),[m*n,r]);
G2    = reshape(diff_y(U_mat,sizeU),[m*n,r]);
mask_mat = reshape(mask,[m*n,p]);
S = randn(m*n,p);
S = S.*(1-mask_mat);
Y1 = zeros(m*n,p);
Y2 = zeros([m*n,r]);
Y3 = Y2;

mu = 1e-1; 
rho = 1.05; 
% mu = 1e-4; 
% rho = 1.1;
%mu = 10; rho = 1.1;   
max_mu = 1e8;

Eny_x   = ( abs(psf2otf([+1; -1], sizeU)) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], sizeU)) ).^2  ;
determ  =  Eny_x + Eny_y;

iter = 0;
total_svd = 0;
converged = false;
flag = 0;
total_nuclear_old = 1e6;
while ~converged      
    X_old = X;
    iter = iter + 1;
    %% update S
    S = D - X + (1/mu)*Y1;
    S = S.*(1-mask_mat);
     %% update V_mat
    tmp = D - S + (1/mu)*Y1;
    if flag ==0
        [U,~,V] = svd(tmp'*U_mat); % (p*mn) * (mn*r) = p*r 
        V_mat = U(:,1:r)*V';% p*r
    end
    %% update U_mat
    U_tmp = tmp*V_mat;
    % update U_mat
    diffT_p  = diff_xT(mu*G1-Y2,sizeU)+diff_yT(mu*G2-Y3,sizeU);
    numer1   = reshape( diffT_p + mu*U_tmp(:), sizeU);
    u        = real( ifftn( fftn(numer1) ./ (mu*determ + mu) ) );
    U_mat    = reshape(u,[m*n,r]);
    X = U_mat*V_mat';
    % update G1,G2
    tmpG1 = reshape(diff_x(U_mat,sizeU),[m*n,r])+Y2/mu;
    tmpG2 = reshape(diff_y(U_mat,sizeU),[m*n,r])+Y3/mu;
    total_nuclear = 0;
    for i=1:r
        [u,s,v] = svd(reshape(tmpG1(:,i),[m,n]));
        diagS = diag(s);
        total_nuclear = total_nuclear + sum(diagS);
        svp = length(find(diagS >1/mu));
        if svp > 0
            shrink = u(:,1:svp) * diag(diagS(1:svp) - 1/mu) * v(:, 1:svp)'; 
            G1(:,i) = shrink(:);
        end
        [u,s,v] = svd(reshape(tmpG2(:,i),[m,n]));
        diagS = diag(s);
        total_nuclear = total_nuclear + sum(diagS);
        svp = length(find(diagS >1/mu));
        if svp > 0
            shrink = u(:,1:svp) * diag(diagS(1:svp) - 1/mu) * v(:, 1:svp)'; 
            G2(:,i) = shrink(:);
        end
    end
    Y1 = Y1 + mu*(D - X - S);
    Y2 = Y2 + mu*(reshape(diff_x(U_mat,sizeU),[m*n,r])-G1);
    Y3 = Y3 + mu*(reshape(diff_y(U_mat,sizeU),[m*n,r])-G2);
    mu = min(mu*rho, max_mu);
    if mu == max_mu
        flag = 1;
    end
    total_svd = total_svd + 1;
    %% stop Criterion   
    stopCriterion = norm(X_old-X)/normD;
    if total_nuclear_old >=total_nuclear
        total_nuclear_old = total_nuclear;
        X_best = reshape(X,[m,n,p]);
    end
    
    if stopCriterion < tol && iter >200
        converged = true;
    end    
    
    if iter == 1 ||mod(total_svd,40) ==0
        disp(['   iter = ',num2str(total_svd),', nuclear =',num2str(total_nuclear), ', mu =',num2str(mu), ', stopCriterion is: ',num2str(stopCriterion)]);
    end   
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
X  = reshape(X,[m,n,p]);
