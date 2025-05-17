function [X,obj,err,iter] = DDTCTV_RPCA(Ten,opts)

[m,n,p] = size(Ten);
dim = size(Ten);
D = reshape(Ten,[m*n,p]);
lambda = 2/sqrt(max(m,n));
norm_two = lansvd(D, 1, 'L');
norm_inf = norm( D(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
mu = 1.25/dual_norm;
max_mu = mu * 1e7;
rho = 1.1;          % this one can be tuned
max_iter = 300;
DEBUG = 1;
is_sp = 1;
tol = 1e-5;
eps = 1e-8;
if ~exist('opts', 'var')
    opts = [];
end    

Eny_x   = ( abs(psf2otf([+1; -1], [m,n,p])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [m,n,p])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [n,p,m])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y; 
if is_sp == 1
    determ  =  Eny_x + Eny_y + Eny_z;
    lambda = 3/sqrt(max(m,n));
end

if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end
if isfield(opts, 'is_sp');       is_sp = opts.is_sp;          end
if isfield(opts, 'lambda');      lambda = opts.lambda;        end

X = zeros(dim);
E = zeros(dim);
Y1 = E;
Y2 = zeros(dim);
Y3 = zeros(dim);
Y4 = zeros(dim);

for iter = 1 : max_iter
%     tic
    Xk = X;
    Ek = E;
    % update G_i  prox_ddttn
    [G1,tnnG1] = prox_ddtnn(reshape(diff_x(X,dim),dim)+Y2/mu,1/mu);
    [G2,tnnG2] = prox_ddtnn(reshape(diff_y(X,dim),dim)+Y3/mu,1/mu);
    if is_sp == 1
        [G3,tnnG3] = prox_ddtnn(reshape(diff_z(X,dim),dim)+Y4/mu,1/mu);
    end
    % update X  FFT
    diffT_p  = diff_xT(mu*G1-Y2,dim)+diff_yT(mu*G2-Y3,dim);
    if is_sp == 1
        diffT_p  = diffT_p + diff_zT(mu*G3-Y4,dim);
    end
    diffT_p = reshape(diffT_p + mu*(Ten(:)-E(:)+Y1(:)/mu),dim);
    x       = real( ifftn( fftn(diffT_p) ./ (mu*determ + mu + eps) ) );
    X       = reshape(x,dim);
    % update E
    temp_T = Ten-X+Y1/mu;
    E = max(temp_T - lambda/mu, 0);
    E = E+min(temp_T + lambda/mu, 0);
    dY = Ten-X-E;    
    chgX = max(abs(Xk(:)-X(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chg = max([chgX chgE max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
%             Xhat = Lateral2Frontal(Xhat); % each lateral slice is a channel of the image          
            obj = tnnG1+tnnG2;
            if is_sp == 1
                obj = tnnG1+tnnG2+tnnG3;
            end
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y1 = Y1 + mu*dY;
    Y2 = Y2 + mu*(reshape(diff_x(X,dim),dim)-G1);
    Y3 = Y3 + mu*(reshape(diff_y(X,dim),dim)-G2);
    if is_sp == 1
        Y4 = Y4 + mu*(reshape(diff_z(X,dim),dim)-G3);
    end
    mu = min(rho*mu,max_mu);    
end
obj = tnnG1+tnnG2;
err = norm(dY(:));

 