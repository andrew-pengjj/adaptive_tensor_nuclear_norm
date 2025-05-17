clear all;clc
addpath(genpath(cd))
filename = 'PaviaU.mat';
load(filename)
X =data;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
dim =[n1,n2,n3];
Xn = X;
%Xn = imnoise(Xn, 'salt & pepper', 0.2);
Xn = imnoise(Xn, 'gaussian', 0, 0.2);
%% Switch Turn on/off
switch_flag.TNN = 1;
switch_flag.CTV = 1;
switch_flag.TCTVF = 1;
switch_flag.TCTVD = 1;
switch_flag.ATNN = 1;
%% Parameter Setting
weight_rpca= 0.8;
rk = 6;
tnn_para.lambda = weight_rpca/sqrt(max(n1,n2)*n3);
atnn_para.lambda = 2*weight_rpca/sqrt(max(n1,n2));
atnn_para.rk = rk;
ctv_para.lambda = 3*weight_rpca/sqrt(n1*n2);
tctvf_para.lambda = weight_rpca/sqrt(prod(dim)/min(dim(1),dim(2)));
tctvd_para.lambda = weight_rpca/sqrt(max(n1,n2));
tctvd_para.transform_matrices = dct(eye(n3));
[MPSNR, MSSIM, ERGAS, Times] = DenoisingList(Xn,X,switch_flag,tnn_para,atnn_para,ctv_para,tctvf_para,tctvd_para);





