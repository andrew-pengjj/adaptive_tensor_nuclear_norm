function [Lmse, RunTime]=GetTcResult(dim,r3rank,tubal_rank,sampling_rate)
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
    m          = round(prod(dim)*sampling_rate);
    sort_dim   = randperm(prod(dim));
    Omega      = sort_dim(1:m); % sampling pixels' index
    Oten        = zeros(dim);
    Oten(Omega) = RLten(Omega); % observed Img
    mask        = zeros(dim);
    mask(Omega) = 1; 
    methodName = {'TNN', 'ATNN','TCTV','ATCTV'};
    Lmse = zeros(4,1);
    RunTime = zeros(4,1);
    %% Run TNN
    it = 1;
    disp(['Running ',methodName{it}, ' ... ']); 
    tic
    OutX = TNN_TC(Oten,mask);
    RunTime(it) = toc;
    X = reshape(OutX,[n1*n2,n3]);
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    %% Run ATNN
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    tic
    OutX = ATNN_TC(Oten, mask, r3);
    RunTime(it) = toc;
    X = reshape(OutX,[n1*n2,n3]);
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    %% Run TCTV
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    tic
    OutX = TCTV_TC(Oten, Omega);
    RunTime(it) = toc;
    X = reshape(OutX,[n1*n2,n3]);
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
    %% Run ATCTV
    it = it+1;
    disp(['Running ',methodName{it}, ' ... ']); 
    clear opts
    tic
    OutX = ATCTV_TC(Oten, mask, r3);
    RunTime(it) = toc;
    X = reshape(OutX,[n1*n2,n3]);
    Lmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
end