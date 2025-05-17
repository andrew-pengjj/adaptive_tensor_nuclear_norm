clear
close all

n1 = 50;
n2 = 50;
n3 = 50;
dim = [n1,n2,n3];

%% parameter settings
R  = 10; % tubal rank
r3 = 10;
sparsity = 0.3;

L = orth(randn(n3,r3))';
smooth_flag = 1;
out = GenerateLRT([n1,n2,r3],R,smooth_flag);
RLten =  COMT(out,L');
RLmat = reshape(RLten,[n1*n2,n3]);
NormD = norm(RLten(:));
realE = GenerateST(n1,n2,n3,round(sparsity*prod(dim)));
Oten = RLten + realE;
Omat = reshape(Oten,[n1*n2,n3]);

methodName = {'NN', 'TNN', 'ATNN','CTV', 'TCTV'};

it = 1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
[A_hat,E_hat,iter] = RPCA(Omat);
run_time(it) = toc;
nmse(it) = norm(A_hat(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');

it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
lambda = 1/sqrt(max(n1,n2));
X = TNN_RPCA(Oten,lambda);
run_time(it) = toc;
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');


it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
X = ATNN_RPCA(Oten, r3);
run_time(it) = toc;
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');

it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
X = CTV_RPCA(Oten);
run_time(it) = toc;
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');


it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
opts = [];
opts.rho = 1.25;
opts.directions = [1,2,3];
weight = max(2-sparsity*2,1+r3/n3*2);
opts.lambda = weight/sqrt(prod(dim)/min(dim(1),dim(2)));
tic
X = TCTV_TRPCA(Oten,opts);
run_time(it) = toc;
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');

fprintf('================== QA Results: RPCA =====================\n');
fprintf(' %8.8s      %5.5s      %5.5s   \n','Method', 'ERROR', 'TIME');

for i = 1:length(methodName)
    fprintf(' %8.8s      %5.5f      %5.5f   \n',methodName{i}, nmse(i), run_time(i));
end