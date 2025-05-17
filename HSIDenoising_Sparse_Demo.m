clear all;clc
addpath(genpath(cd))
filename = 'PaviaU.mat';
load(filename)
X =data;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
dim =[n1,n2,n3];
Xn = X;
Xn = imnoise(Xn, 'salt & pepper', 0.4);
%% Switch Turn on/off
switch_flag.TNN = 0;
switch_flag.CTV = 1;
switch_flag.TCTVF = 0;
switch_flag.TCTVD = 0;
switch_flag.ATNN = 0;
%% Parameter Setting
weight_rpca= 1;
rk = 25;
tnn_para.lambda = weight_rpca/sqrt(max(n1,n2)*n3);
atnn_para.lambda = weight_rpca/sqrt(max(n1,n2));
atnn_para.rk = rk;
ctv_para.lambda = 3*weight_rpca/sqrt(n1*n2);
tctvf_para.lambda = weight_rpca/sqrt(prod(dim)/min(dim(1),dim(2)));
tctvd_para.lambda = weight_rpca/sqrt(max(n1,n2));
tctvd_para.transform_matrices = dct(eye(n3));
[MPSNR, MSSIM, ERGAS, Times] = DenoisingList(Xn,X,switch_flag,tnn_para,atnn_para,ctv_para,tctvf_para,tctvd_para);






