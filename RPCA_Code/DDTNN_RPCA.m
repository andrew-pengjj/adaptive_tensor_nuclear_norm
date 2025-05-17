function [X,obj,err,iter] = DDTNN_RPCA(Ten,opts)

[m,n,p] = size(Ten);
dim = size(Ten);
lambda = 1/sqrt(max(m,n)); 
D = reshape(Ten,[m*n,p]);
%normD = norm(D(:));
norm_two = lansvd(D, 1, 'L');
norm_inf = norm( D(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
mu = 1.25/dual_norm;
max_mu = mu * 1e7;
rho = 1.25;          % this one can be tuned
max_iter = 300;
DEBUG = 1;
tol = 1e-6;
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

X = zeros(dim);
E = zeros(dim);
Y = E;

for iter = 1 : max_iter
%     tic
    Xk = X;
    Ek = E;
    % update X  prox_ddttn
    [X,tnnX] = prox_ddtnn(Ten-E+Y/mu,1/mu); 
    % update E
    temp_T = Ten-X+Y/mu;
    E = max(temp_T - lambda/mu, 0);
    E = E+min(temp_T + lambda/mu, 0);
    dY = Ten-X-E;    
    chgX = max(abs(Xk(:)-X(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chg = max([chgX chgE max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
%             Xhat = Lateral2Frontal(Xhat); % each lateral slice is a channel of the image          
            obj = tnnX;
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);    
end
obj = tnnX;
err = norm(dY(:));

 