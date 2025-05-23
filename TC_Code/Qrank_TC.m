function [X,obj,err,iter] = lrtc_tqn(M,omega,opts)

% Solve the Low-Rank Tensor Completion (LRTC) based on Tensor Nuclear Norm (TNN) problem by M-ADMM
%
% min_X ||X||_*, s.t. P_Omega(X) = P_Omega(M)
%
% ---------------------------------------------
% Input:
%       M       -    d1*d2*d3 tensor
%       omega   -    index of the observed entries
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    d1*d2*d3 tensor
%       err     -    residual
%       obj     -    objective function value
%       iter    -    number of iterations
%
% version 1.0 - 25/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 
% References:
% Canyi Lu, Jiashi Feng, Zhouchen Lin, Shuicheng Yan
% Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements
% International Joint Conference on Artificial Intelligence (IJCAI). 2018


tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end
if isfield(opts, 'rk');          rk = opts.rk;          end

dim = size(M);
X = zeros(dim);
X(omega) = M(omega);
E = zeros(dim);
Y = E;

iter = 0;



for iter = 1 : max_iter
%     tic
    Xk = X;
    Ek = E;
    % update X
    [X,tnnX] = prox_tqn(-E+M+Y/mu,1/mu,rk); 
    % update E
    E = M-X+Y/mu;
    E(omega) = 0;
 
    dY = M-X-E;    
    chgX = max(abs(Xk(:)-X(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chg = max([chgX chgE max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 1) == 0
            obj = tnnX;
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err) ', psnr=' num2str(psnr)]); 
            
            
        end
        
        if iter == 1 || mod(iter, 10) == 0
            %save 'RES_MAT\TQN\Video_Xhat_0_5.mat' Xhat;
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);    
    
    
%     toc

    
end
obj = tnnX;
err = norm(dY(:));

 