function [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc,err_valid]=rawpmf(train_vec,probe_vec,epsilon,lambdau,lambdav,momentum, ...
    maxepoch,num_feat,w1_P1,w1_M1,w1_P1_inc,w1_M1_inc)
%this pmf is the most basic version, not applying g to (U^T)V and not
%normalising the ratings

rand('state',0); 
randn('state',0); 

%epsilon= Learning rate x 100000 (note for batch gradient descent we use
%learning-rate/mini-batch-size as the step size.
% lambdau,lambdav=Regularization parameters
epoch=1;
  
ratings_test = double(probe_vec(:,3));
 
pairs_tr = length(train_vec); % number of training data 
pairs_pr = length(probe_vec); % number of validation data 

num_m=17770; 
num_p=480189;
numbatches=991;

if ~exist('w1_M1','var') 
  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
end

if ~exist('w1_P1','var')
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
end

if ~exist('w1_M1_inc','var')
  w1_M1_inc = zeros(num_m, num_feat); %increments to parameters for GD
end
if ~exist('w1_P1_inc','var')
  w1_P1_inc = zeros(num_p, num_feat); %increments to parameters for GD
end

err_train=zeros(maxepoch,1);
err_valid=zeros(maxepoch,1);
N=100000; % number training triplets per batch 

for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:); %randomly permute the training data
  %note that the order of the data may be important, as we are updating
  %observation by observation. To avoid this sampling bias, we mix the data
  %randomly.
  clear rr 

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);

    aa_p   = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),1)); %N rows for user col
    aa_m   = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),2)); %N rows for movie col
    rating = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),3)); %N rows of ratings

    %rating = rating-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    %ff = find(pred_out>5); pred_out(ff)=5; 
    %ff = find(pred_out<1); pred_out(ff)=1; %need for vb init pmf
    %f = sum( (pred_out - rating).^2 + ...
    %    lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2))); %value of minimising objective

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat); % num_feat copies of 2*(pred_out - rating) concatenated
    %ie. is a N by num_feat matrix
    Ix_m=IO.*w1_P1(aa_p,:) + 2*lambdav*w1_M1(aa_m,:); %each row=gradient for Vj (j=1 to N)
    Ix_p=IO.*w1_M1(aa_m,:) + 2*lambdau*w1_P1(aa_p,:); %each row = gradient for Ui (i=1 to N)

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);
    if batch~=numbatches
            for ii=1:N   %initialise gradients ie. add up all the above for each user and movie
                %cannot vectorise as vectorisation does things simultaneously, which is implausible
                dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
                dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
            end
    else    for ii=1:(pairs_tr-N*(batch-1))
                dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
                dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
            end
    end
    %%%% Update movie and user features %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc - epsilon*dw1_M1/N; %note division by N
    w1_M1 =  w1_M1 + w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc - epsilon*dw1_P1/N; %note division by N
    w1_P1 =  w1_P1 + w1_P1_inc;
  end 

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  
  probe_rat = pred(w1_M1,w1_P1,probe_vec,0);
       
  temp = (ratings_test - probe_rat).^2;
  err = sqrt(sum(temp)/pairs_pr);

  err_valid(epoch)=err;

  fprintf(1, 'epoch %4i batch %4i Average Test RMSE %6.4f \n', epoch,batch,err);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %if (rem(epoch,save_epoch))==0
  %   save /alt/applic/user-maint/hjk42/rawpmf_weights_and_errors30 w1_M1 w1_P1 w1_M1_inc w1_P1_inc err_valid
  %end

end 
end