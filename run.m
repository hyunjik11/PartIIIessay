%%%%%%%%%% REMEMBER TO SAVE!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1==1 %pentopia
    load all_data
    tic
    [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=pmf(trainU,probeU,50,0.01,0.001,0.9, ...
    30,60);
    toc
end

if 1==0 %oval
    load all_data
    tic
    [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainU,probeU,50,0.01,0.001,0.9,30,60);
    toc
    %save /alt/applic/user-maint/hjk42/rawpmf60errors err_valid
end

if 1==0 %otter 
load all_data
tic
[U,V,Psi,sigma,tau,overall_err]=rawvb(trainU,probeU,30,60);
toc
%save /alt/applic/user-maint/hjk42/rawvb60errors err_valid
end

if 1==0 %cave
    load all_data
    load rawvb_random60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=rawbayespmf(trainM,probeU, R, ...
    30,60,8,U,V);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init60_8_30epochs w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end

if 1==0 %tomtom
    load all_data
    load rawpmf60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=rawbayespmf(trainM,probeU, R, ...
    30,60,8,w1_P1,w1_M1);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_rawpmf_init60_8_30epochs w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end

if 1==0 %tangram
    load all_data
    load pmf_weights_and_errors60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=bayespmf(trainM,probeU, R, ...
    30,60,8,w1_P1,w1_M1);
    toc
    %save /alt/applic/user-maint/hjk42/bayespmf60_8_30epochs w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end

if 1==0 %pentopia
    load all_data
    load rawvb_random30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=rawbayespmf(trainM,probeU, R, ...
    30,30,8,U,V);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init30_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end

if 1==0 %stadium
    load all_data
    load rawpmf_weights_and_errors30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=rawbayespmf(trainM,probeU, R, ...
    30,30,8,w1_P1,w1_M1);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_rawpmf_init30_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end

if 1==0 %hashi
    load all_data
    load pmf_weights_and_errors30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m,probe_rat_all,counter_prob]=bayespmf(trainM,probeU, R, ...
    30,30,8,w1_P1,w1_M1);
    toc
    %save /alt/applic/user-maint/hjk42/bayespmf_weights_and_errors30_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m probe_rat_all counter_prob
end
    

if 1==0
    load all_data
    load pmf_weights_and_errors30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=bayespmf(trainM,probeM, R,30,30,4,w1_P1,w1_M1);
    toc
    save /alt/applic/user-maint/hjk42/bayespmf_weights_and_errors30_4 w1_P1_sample w1_M1_sample overall_err
end

if 1==0 
    load all_data
    load bayespmf60_8
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=bayespmf(trainM,probeM, R, ...
    50,60,8,w1_P1_sample,w1_M1_sample,mu_u,mu_m,alpha_u,alpha_m);
    toc
    %save /alt/applic/user-maint/hjk42/bayespmf60_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m
end

if 1==0 
    load all_data
    load rawbayespmf_rawpmf_init60_8
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=rawbayespmf(trainM,probeM,R,50,M,N,60,8,w1_P1_sample,w1_M1_sample,mu_u,mu_m,alpha_u,alpha_m);
    toc
    % save /alt/applic/user-maint/hjk42/rawbayespmf_rawpmf_init60_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m
end

if 1==0
    load all_data
    load rawbayespmf_vb_init60_8
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=rawbayespmf(trainM,probeM,R,50,M,N,60,8,w1_P1_sample,w1_M1_sample,mu_u,mu_m,alpha_u,alpha_m);
    toc
    % save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init60_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m
end

if 1==0 
    load all_data
    load pmf_weights_and_errors60
    load R
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=bayespmf(trainM,probeM, R, ...
    50,60,8,w1_P1,w1_M1);
    %save /alt/applic/user-maint/hjk42/bayespmf60_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m
end

if 1==0  
    load all_data
    load rawvb_random60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err,mu_u,mu_m,alpha_u,alpha_m]=rawbayespmf(trainM,probeM,R,50,M,N,60,8,U,V);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init60_8 w1_P1_sample w1_M1_sample overall_err mu_u mu_m alpha_u alpha_m
end

if 1==0  
    load all_data
    load rawvb_random30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,30,8,U,V);
    toc
    %save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init30_8 w1_P1_sample w1_M1_sample overall_err
end

if 1==0
load all_data
tic
[w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=pmf2(trainM,probeM,50,0.001,0.0001,0.9,10,30);
toc
save /alt/applic/user-maint/hjk42/pmfsigmoid2_30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
end

if 1==0 
load all_data
tic
[w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=pmf(trainM,probeM,50,0.001,0.0001,0.9,10,30);
toc
save /alt/applic/user-maint/hjk42/pmf2_30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
end

if 1==0 
load all_data
tic
[w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainM,probeM,50,0.001,0.0001,0.9,10,30);
toc
save /alt/applic/user-maint/hjk42/rawpmf2_weights_and_errors30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
end

if 1==0  
    load all_data
    load rawvb_random60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,60,32,U,V);
    toc
    save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init60 w1_P1_sample w1_M1_sample overall_err
end

if 1==0 
    load all_data
    load rawbayespmf_rawpmf_init30
    tic
    [U,V,Psi,sigma,tau,overall_err]=rawvb(trainU,probeU,30,30,w1_P1_sample,w1_M1_sample);
    toc
    save /alt/applic/user-maint/hjk42/rawvb_bayespmf_init30 U V Psi sigma tau overall_err
end

if 1==0 
    load all_data
    load rawpmf60
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,60,32,w1_P1,w1_M1);
    toc
    save /alt/applic/user-maint/hjk42/rawbayespmf_rawpmf_init60 w1_P1_sample w1_M1_sample overall_err
end


if 1==0 
    load all_data
    load rawvb_random30
    load R
    tic
    [w1_P1_sample,w1_M1_sample,overall_err]=rawbayespmf(trainM,probeM,R,30,M,N,30,32,U,V);
    toc
    save /alt/applic/user-maint/hjk42/rawbayespmf_vb_init30 w1_P1_sample w1_M1_sample overall_err
end

if 1==0 
    load all_data
    tic
    [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(trainM,probeM,50,0.01,0.001,0.9,6,60);
    toc
    save /alt/applic/user-maint/hjk42/rawpmf60 w1_P1 w1_M1 w1_P1_inc w1_M1_inc err_valid
end

if 1==0 
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

if 1==0 
    load all_data
    tic
    [U,V,Psi,sigma,tau,overall_err]=rawvb(trainU,probeU,9,60);
    toc
    save /alt/applic/user-maint/hjk42/rawvb_random60 U V overall_err Psi sigma tau
end

if 1==0 
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