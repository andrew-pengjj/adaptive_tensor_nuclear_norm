function [MPSNR, MSSIM, ERGAS, Times] = DenoisingList(NoiseData,CleanX,switch_flag,tnn_para,atnn_para, ctv_para,tctvf_para,tctvd_para)
[n1,n2,n3] = size(NoiseData);
MethodName = {'Nosiy','TNN','ATNN','CTV','TCTV-DFT','TCTV-DCT'};
MPSNR = zeros(length(MethodName),1);
MSSIM = zeros(length(MethodName),1);
ERGAS = zeros(length(MethodName),1);
Times = zeros(length(MethodName),1);
it = 1;
[MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX, NoiseData);

it = it + 1;
if switch_flag.TNN == 1
    if exist('tnn_para','var')
        if isfield(tnn_para,'lambda')
            lambda = tnn_para.lambda;
        else
            lambda = 1/sqrt(max(n1,n2)*n3);
        end
    end
    
    opts = [];
    opts.DEBUG = 1; 
    fprintf(' ============= %s =============\n',MethodName{it});
    tic;
    TNN_Res = TNN_RPCA(NoiseData,lambda,opts);
    Times(it) = toc;
    [MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX, TNN_Res);
end

it = it + 1;
if switch_flag.ATNN == 1
    fprintf(' ============= %s =============\n',MethodName{it});
    opts.DEBUG = 1;
    opts.lambda = atnn_para.lambda;
    opts.r = atnn_para.rk;
    tic;
    ATNN_Res = ATNN_RPCA(NoiseData,opts);
    Times(it) = toc;
    [MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX,ATNN_Res);
end

it = it + 1;
if switch_flag.CTV == 1
    fprintf(' ============= %s =============\n',MethodName{it});
    opts = [];
    if exist('ctv_para','var')
        opts.lambda = ctv_para.lambda;
        if isfield(ctv_para,'weight')
            opts.weight = ctv_para.weight;
        else
            opts.weight = 1;
        end
    end
    
    tic
    CTV_Res = CTV_RPCA(NoiseData,opts);
    Times(it) = toc;
    [MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX,CTV_Res);
end

it = it + 1;
if switch_flag.TCTVF == 1
    fprintf(' ============= %s =============\n',MethodName{it});
    opts = [];
    opts.rho = 1.25;
    opts.directions = [1,2,3];
    opts.lambda = tctvf_para.lambda;
    tic
    TCTV_Res = TCTV_TRPCA(NoiseData,opts);
    Times(it) = toc;
    [MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX,TCTV_Res);
end

it = it + 1;
if switch_flag.TCTVD == 1
    fprintf(' ============= %s =============\n',MethodName{it});
    opts = [];
    opts.rho = 1.25;
    opts.directions = [1,2,3];
    opts.lambda = tctvd_para.lambda;
    opts.transform_matrices = tctvd_para.transform_matrices;
    opts.transform= 'DCT';
    tic
    TCTV_Res = TCTV_TRPCA(NoiseData,opts);
    Times(it) = toc;
    [MPSNR(it),MSSIM(it),ERGAS(it)] = msqia(CleanX,TCTV_Res);
end

fprintf('================== QA Results =====================\n');
fprintf(' %8.8s    %5.5s    %5.5s    %5.5s    %5.5s \n',...
    'Method', 'MPSNR', 'MSSIM', 'ERGAS', 'Time');
for i = 1:length(MethodName)
    fprintf(' %8.8s   %5.3f    %5.3f    %5.3f    %5.3f   \n',...
        MethodName{i}, MPSNR(i), MSSIM(i), ERGAS(i),Times(i));
end


