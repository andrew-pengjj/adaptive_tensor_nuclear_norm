addpath(genpath(cd))
clear

% pic_name = './testimg.jpg';
%pic_name = '4.2.06.tiff';
pic_name = '3096.jpg';
X = double(imread(pic_name));

X = X/255;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
dim =[n1,n2,n3];
Xn = X;
rhos = 50/255;
%Xn = imnoise(Xn, 'salt & pepper', rhos);
Xn = imnoise(Xn, 'gaussian', 0, rhos);

% ind = find(rand(n1*n2*n3,1)<rhos);
% Xn(ind) = rand(length(ind),1);

%% TNN-TRPCA
opts.mu = 1e-4;
opts.tol = 1e-5;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 1;
[n1,n2,n3] = size(Xn);
lambda = 1/sqrt(max(n1,n2)*n3);
[Xhat,E,err,iter] = TNN_RPCA(Xn,lambda,opts);
Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
psnr = PSNR_c(X*255,Xhat*255,n1,n2)

opts = [];
opts.rho = 1.25;
opts.directions = [1,2,3];
opts.transform= 'DCT';
weight = 1;%max(2-rhos*2,1);
opts.lambda = weight/sqrt(prod(dim)/min(dim(1),dim(2)));
tic
Xhat1 = TCTV_TRPCA(Xn,opts);
Xhat1 = max(Xhat1,0);
Xhat1 = min(Xhat1,maxP);
psnr1 = PSNR_c(X*255,Xhat1*255,n1,n2)

opts.r =3;
weight = 1;%min(2-rhos,1+opts.r/n3);
opts.lambda = 2*weight/sqrt(max(n1,n2));
Xhat2 = ATCTV_RPCA(Xn,opts);
Xhat2 = max(Xhat2,0);
Xhat2 = min(Xhat2,maxP);
psnr2 = PSNR_c(X*255,Xhat2*255,n1,n2)


Xhat3 = DDTNN_RPCA(Xn);
Xhat3 = max(Xhat3,0);
Xhat3 = min(Xhat3,maxP);
psnr3 = PSNR_c(X*255,Xhat3*255,n1,n2)


opts.weight = 0.5;
opts.rho = 1.25;
Xhat4 = DDTCTV_RPCA(Xn,opts);
Xhat4 = max(Xhat4,0);
Xhat4 = min(Xhat4,maxP);
psnr4 = PSNR_c(X*255,Xhat4*255,n1,n2)



% [~,Xhat3] = CBM3D(1, Xn, rhos*255);
% psnr3 = PSNR_c(X*255,Xhat3*255,n1,n2)

figure(1)
subplot(2,3,1)
imshow(X/max(X(:)));title('clean')
subplot(2,3,2)
imshow(Xhat/max(Xhat(:)));title(['TNN: ',num2str(psnr)])
subplot(2,3,3)
imshow(Xhat1/max(Xhat1(:)));title(['TCTV: ',num2str(psnr1)])
subplot(2,3,4)
imshow(Xhat2/max(Xhat2(:)));title(['ATCTV: ',num2str(psnr2)])
subplot(2,3,5)
imshow(Xhat3/max(Xhat3(:)));title(['Qrank-TNN: ',num2str(psnr3)])
subplot(2,3,6)
imshow(Xhat4/max(Xhat4(:)));title(['Qrank-TCTV: ',num2str(psnr4)])




