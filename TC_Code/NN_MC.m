%% solve the problem 
%              min_X ||D_x(X)||_*+||D_y(X)||_*
%                                   s.t.  Y_Omega= X_Omega
%                          ===============================
%              min_X ||X1||_*+||X2||_*
%                            s.t.  Y_Omega= X_Omega
%                                  D_x(X)=X1 
%                                  D_y(X)=X2 
%                          ===============================                       
%         D is difference operator,T is difference tensor,T is known
%  ------------------------------------------------------------------------


function output_image =NN_MC(D,Omega)
tol     = 1e-4;
maxIter = 100;
rho     = 1.5;
[M,N] = size(D);
normD   = norm(D,'fro');
% initialize
%norm_two = lansvd(D, 1, 'L');
norm_two = svds(D, 1, 'L');
norm_inf = norm( D(:), inf);
dual_norm = max(norm_two, norm_inf);

mu = 0.001;%1.25/dual_norm%1.25/norm_two % this one can be tuned
max_mu = mu * 1e7;
%% Initializing optimization variables
X  = D;
E  = zeros(M,N);
M1 = E; 
% main loop
iter = 0;
tic
while iter<maxIter
    iter          = iter + 1;   
    %% -Updata X1,X2,X3
    [u,s,v] = svd(D-E+M1/mu,'econ');
    X      = u*softthre(s,1/mu)*v';
    %% -Update E
    E  = D-X+M1/mu; 
    E(Omega) = 0;
        %% stop criterion  
    leq = D -X -E;
    stopC = norm(leq,'fro')/normD;
    if mod(iter,10) ==0
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e')  ...
            ',||Y-X-E||_F/||Y||_F=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol
        break;
    else
        M1 = M1 + mu*leq;
        mu = min(max_mu,mu*rho); 
    end 
end
X(Omega)=D(Omega);
output_image = X;

end