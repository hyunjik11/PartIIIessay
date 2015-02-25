if 1==0
load all_data
tic
pmf(trainM,probeM,50,0.01,0.001,0.9,5,5,991,17770,480189,60);
toc
end

if 1==1
load all_data
load R
load pmf_weights_and_errors60
tic
bayespmf(trainM,probeM,R,30,M,N,60,32,w1_P1,w1_M1);
toc
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