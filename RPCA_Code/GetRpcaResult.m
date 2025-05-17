function [Lmse, Smse, RunTime]=GetRpcaResult(dim,r3rank,tubal_rank,sparsity)
    n1 = dim(1);
    n2 = dim(2);
    n3 = dim(3);
    r3 = r3rank;
    R  = tubal_rank;
    %% generate data
    L = orth(randn(n3,r3))';
    smooth_flag = 1;
    out = GenerateLRT([n1,n2,r3],R,smooth_flag);
    RLten =  COMT(out,L');
    RLmat = reshape(RLten,[n1*n2,n3]);
    realE = GenerateST(n1,n2,n3,round(sparsity*prod(dim)));
    Oten = RLten + realE;
    %Omat = reshape(Oten,[n1*n2,n3]);
    methodName = {'TNN', 'ATNN','TCTV','ATCTV'};
    Lmse = zeros(4,1);
    Smse = zeros(4,1);
    RunTime = zeros(4,1);
    %% Run TNN
    it = 1;
    disp(['Running ',methodName{it}, ' ... ']); 
    tic
    lambda = 1/sqrt(max(n1,n2));
    X = TNN_RPCA(Oten,lambda);
    RunTime(it) = toc;
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    Smse(it) = norm(Oten(:)-X(:)-realE(:),'fro')/norm(realE(:),'fro');
    %% Run ATNN
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    tic
    X = ATNN_RPCA(Oten, r3);
    RunTime(it) = toc;
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    Smse(it) = norm(Oten(:)-X(:)-realE(:),'fro')/norm(realE(:),'fro');
    %% Run TCTV
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    opts = [];
    opts.rho = 1.25;
    opts.directions = [1,2,3];
    weight = max(2-sparsity*2,1+r3/n3*2);
    opts.lambda = weight/sqrt(prod(dim)/min(dim(1),dim(2)));
    tic
    X = TCTV_TRPCA(Oten,opts);
    RunTime(it) = toc;
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    Smse(it) = norm(Oten(:)-X(:)-realE(:),'fro')/norm(realE(:),'fro');
    %% Run ATCTV
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    clear opts
    opts.r = r3;
    weight = max(2-sparsity*2,1+r3/n3*2);
    opts.lambda = 2*weight/sqrt(max(n1,n2));
    tic
    X = ATCTV_RPCA(Oten,opts);
    RunTime(it) = toc;
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    Smse(it) = norm(Oten(:)-X(:)-realE(:),'fro')/norm(realE(:),'fro');
end