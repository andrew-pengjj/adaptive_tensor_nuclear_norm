clear
close all

n1 = 50;
n2 = 50;
n3 = 50;
dim = [n1,n2,n3];

%% parameter settings
R  = 10; % tubal rank
r3 = 5;
sampling_rate = 0.5;
L = orth(randn(n3,r3))';
smooth_flag = 1;
out = GenerateLRT([n1,n2,r3],R,smooth_flag);
RLten =  COMT(out,L');
RLmat = reshape(RLten,[n1*n2,n3]);
NormD = norm(RLten(:));

m          = round(prod(dim)*sampling_rate);
sort_dim   = randperm(prod(dim));
Omega      = sort_dim(1:m); % sampling pixels' index
Oten        = zeros(dim);
Oten(Omega) = RLten(Omega); % observed Img
mask        = zeros(dim);
mask(Omega) = 1; 
Omat = reshape(Oten,[n1*n2,n3]);
Omemat = find(Omat~=0);

methodName = {'NN', 'TNN', 'ATNN', 'CTV','TCTV'};

%% NN_MC
it = 1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
X = NN_MC(Omat,Omemat);
run_time(it) = toc;
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
%% TNN_TC
it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
lambda = 1/sqrt(max(n1,n2));
OutX = TNN_TC(Oten,mask);
run_time(it) = toc;
X = reshape(OutX,[n1*n2,n3]);
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');
%% ATNN_TC
it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
OutX = ATNN_TC(Oten, mask, r3);
run_time(it) = toc;
X = reshape(OutX,[n1*n2,n3]);
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');


%% CTV_MC
it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
OutX = CTV_MC(Oten, mask);
run_time(it) = toc;
X = reshape(OutX,[n1*n2,n3]);
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');

%% TCTV_TC
it = it+1;
disp(['Running ',methodName{it}, ' ... ']); 
tic
OutX = TCTV_TC(Oten, Omega);
run_time(it) = toc;
X = reshape(OutX,[n1*n2,n3]);
nmse(it) = norm(X(:)-RLmat(:),'fro')/norm(RLmat(:),'fro');

fprintf('================== QA Results: TC =====================\n');
fprintf(' %8.8s      %5.5s      %5.5s   \n','Method', 'ERROR', 'TIME');

for i = 1:length(methodName)
    fprintf(' %8.8s      %5.5f      %5.5f   \n',methodName{i}, nmse(i), run_time(i));
end