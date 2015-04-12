if 1==0 %otter
    load all_data
    load rawvb_random30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,30,32,U,V);
    toc
    save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init30 w1_P1_sample w1_M1_sample overall_err
end

if 1==1 %stadium 
    load all_data
    tic
    [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainM,probeM,50,0.01,0.001,0.9,6,60);
    toc
    save /alt/applic/user-maint/hjk42/rawpmf60 w1_P1 w1_M1 w1_P1_inc w1_M1_inc err_valid
end

if 1==0 %hashi
    load all_data
    load rawpmf_weights_and_errors30
    tic
    [U,V,Psi,sigma,tau,overall_err]=rawvb(trainU,probeU,30,30,w1_P1,w1_M1);
    toc
    save /alt/applic/user-maint/hjk42/rawvb_pmf_init30 U V Psi sigma tau overall_err
end

if 1==0 
load all_data
load rawvb_random30
tic
[w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainM,probeM,50,0.01,0.001,0.9,10,30,U,V);
toc
save /alt/applic/user-maint/hjk42/rawpmf_vb_init30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
end

if 1==0 %pentopia
    load all_data
    tic
    [U,V,Psi,sigma,tau,overall_err]=rawvb(trainU,probeU,30,60);
    toc
    save /alt/applic/user-maint/hjk42/rawvb_random60 U V overall_err Psi sigma tau
end

if 1==0 %cave
    load all_data
    load rawpmf_weights_and_errors30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,30,32,w1_P1,w1_M1);
    toc
    save /alt/applic/user-maint/hjk42/rawbayespmf_rawpmf_init30 w1_P1_sample w1_M1_sample overall_err
end

if 1==0
    load all_data
    tic
    rawvb(trainU,probeU,30,30);
    toc
end

if 1==0
    load all_data
    load pmf_weights_and_errors60
    tic
    vb(trainU,probeU,30,60,w1_P1,w1_M1); %this won't work well because w1_P1,w1_M1 predict R-mean_rating
    %whereas vb tries to predict R
    toc
end

if 1==0
    load all_data
    load pmf_weights_and_errors30
    tic
    vb(trainU,probeU,30,30,w1_P1,w1_M1); %this won't work well because w1_P1,w1_M1 predict R-mean_rating
    %whereas vb tries to predict R
    toc
end

if 1==0
    load all_data
    load pmf_weights_and_errors30
    Psi=repmat(eye(30)/30,1,1,M);  
    tic
    vb(trainU,probeU,30,M,N,30,w1_P1,w1_M1,Psi);
    toc
end

if 1==0
    load all_data
    tic
    vb(trainU,probeU,30,M,N,30);
    toc
end


if 1==0
load all_data
tic
[w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainM,probeM,50,0.01,0.001,0.9,8,8,991,17770,480189,30);
toc
save /alt/applic/user-maint/hjk42/rawpmf_weights_and_errors30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
end

if 1==0
load all_data
load R
load pmf_weights_and_errors30
tic
[w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,30,32,w1_P1,w1_M1);
toc
save /alt/applic/user-maint/hjk42/rawbayespmf_weights_and_errors30_32 w1_M1_sample w1_P1_sample overall_err
end

if 0==1
    load all_data
    load R
    tic
    full=svm_em(R,5,1);
    indices=(probeU(:,2)-1)*N+probeU(:,1);
    pairs_pr = length(probeU);
    RMSE=sqrt(sum((full(indices)-probeU(:,3)).^2)/pairs_pr);
    fprintf(1,'Test RMSE %6.4f \n',RMSE);
    toc
end