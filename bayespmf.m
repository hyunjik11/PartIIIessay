% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

function [w1_P1_sample,w1_M1_sample]=bayespmf(train_vec,probe_vec, R, ...
    maxepoch,num_m,num_p,num_feat,n_samples,w1_P1,w1_M1)
%use trainM for train_vec
%use w1_P1,w1_M1 from pmf (not pmf2)
rand('state',0);
randn('state',0);

  epoch=1; 

  iter=0; 

  % Initialize hierarchical priors 
  beta=2; % observation noise (precision) (=alpha in paper)
  mu_u = zeros(num_feat,1);
  mu_m = zeros(num_feat,1);
  alpha_u = eye(num_feat); %capital lambda_u in paper
  alpha_m = eye(num_feat);  %capital lambda_m in paper

  % parameters of Inv-Whishart distribution (see paper for details) 
  WI_u = eye(num_feat); %W_0 in paper 
  b0_u = 2; %beta_0 in paper.
  df_u = num_feat; %v_0 in paper
  mu0_u = zeros(num_feat,1);

  WI_m = eye(num_feat);
  b0_m = 2;
  df_m = num_feat;
  mu0_m = zeros(num_feat,1);
  %for all values above we use separate values for u and m, whereas in paper we use the same value

  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 

  w1_P1_sample = w1_P1; 
  w1_M1_sample = w1_M1; 
  clear w1_P1 w1_M1;
  
  overall_err=zeros(maxepoch,1);

  % Initialization using MAP solution found by PMF. 
  %% Do simple fit

  R=R';
  probe_rat_all = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
  counter_prob=1; 

for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see eqn(14) for details)  
  N = size(w1_M1_sample,1);
  x_bar = mean(w1_M1_sample)'; 
  S_bar = cov(w1_M1_sample); 

  WI_post = inv(inv(WI_m) + N/1*S_bar + ...
            N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
  WI_post = (WI_post + WI_post')/2;

  df_mpost = df_m+N;
  alpha_m = wishrnd(WI_post,df_mpost);   
  mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
  lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
  mu_m = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  N = size(w1_P1_sample,1);
  x_bar = mean(w1_P1_sample)';
  S_bar = cov(w1_P1_sample);

  WI_post = inv(inv(WI_u) + N/1*S_bar + ...
            N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
  WI_post = (WI_post + WI_post')/2;
  df_mpost = df_u+N;
  alpha_u = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
  lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  
  
  %here number of samples is 2

  for gibbs=1:n_samples 
    fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    R=R';
    for mm=1:num_m
       fprintf(1,'movie =%d\r',mm);
       ff = find(R(:,mm)>0);
       MM = w1_P1_sample(ff,:);
       rr = R(ff,mm)-mean_rating;
       covar = inv((alpha_m+beta*(MM'*MM)));
       mean_m = covar*(beta*MM'*rr+alpha_m*mu_m); %equation (12&13)
       lam = chol(covar); lam=lam'; 
       w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
     end

    %%% Infer posterior distribution over all user feature vectors 
     R=R';
     for uu=1:num_p
       fprintf(1,'user  =%d\r',uu);
       ff = find(R(:,uu)>0);
       MM = w1_M1_sample(ff,:);
       rr = R(ff,uu)-mean_rating;
       covar=inv((alpha_u+beta*(MM'*MM)));
       mean_u = covar*(beta*MM'*rr+alpha_u*mu_u);
       lam = chol(inv(alpha_u+beta*(MM'*MM))); lam=lam'; 
       w1_P1_sample(uu,:) = lam*randn(num_feat,1)+mean_u;
     end
   end 

   probe_rat = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   counter_prob=counter_prob+1;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   iter=iter+1;
   overall_err(iter)=err;

  fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);

end 
save /alt/applic/user-maint/hjk42/bayespmf_weights_and_errors60_32 w1_M1_sample w1_P1_sample overall_err
end


