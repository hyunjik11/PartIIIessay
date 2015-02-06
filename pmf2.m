function [w1_P1,w1_M1,w1_P1_inc,w1_M1_inc]=pmf2(train_vec,probe_vec,epsilon,lambda,momentum, ...
    maxepoch,numbatches,num_m,num_p,num_feat,w1_P1,w1_M1,w1_P1_inc,w1_M1_inc)
%this pmf is the most basic version, not applying g to (U^T)V and not
%normalising the ratings
g=@(x) 1/(1+exp(-x));
dg=@(x) exp(-x)/((1+exp(-x))^2);
rand('state',0); 
randn('state',0); 

  %epsilon= Learning rate
  % lambda=Regularization parameter(using same lambda for u and v)

  epoch=1;

  mean_rating = mean(train_vec(:,3)); 
 
  pairs_tr = length(train_vec); % number of training data 
  pairs_pr = length(probe_vec); % number of validation data 

  % num_m=Number of movies 
  % num_p=Number of users 
  % num_feat=Rank 
  if nargin<11
    w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
    w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
    w1_M1_inc = zeros(num_m, num_feat); %increments to parameters for GD
    w1_P1_inc = zeros(num_p, num_feat); %increments to paramenters for GD
  elseif (10<nargin)&&(nargin<13)
    w1_M1_inc = zeros(num_m, num_feat); %increments to parameters for GD
    w1_P1_inc = zeros(num_p, num_feat); %increments to paramenters for GD
  end
    
err_train=zeros(maxepoch,1);
err_valid=zeros(maxepoch,1);

for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:); %randomly permute the training data
  %note that the order of the data may be important, as we are updating
  %observation by observation. To avoid this sampling bias, we mix the data
  %randomly.
  clear rr 

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    N=100000; % number training triplets per batch 

    aa_p   = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),1)); %N rows for user col
    aa_m   = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),2)); %N rows for movie col
    rating = double(train_vec((batch-1)*N+1:min(batch*N,pairs_tr),3)); %N rows of ratings

    rating = (rating-1)/4; % rating scaled to between 0 and 1. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = arrayfun(g,sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2)); %g(U'V)
    %f = sum( (pred_out - rating).^2 + ...
    %   lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2))); %value of minimising objective

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    grad_pred_out=arrayfun(dg,sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2)); %g'(U'V) : N by 1 col vector
    IO = repmat(2*grad_pred_out.*(pred_out - rating),1,num_feat); % num_feat copies of 2g'(U'V)(U'V-R) concatenated
    %ie. is a N by num_feat matrix
    Ix_m=IO.*w1_P1(aa_p,:) + 2*lambda*w1_M1(aa_m,:); %each row=gradient for Vj (j=1 to N)
    Ix_p=IO.*w1_M1(aa_m,:) + 2*lambda*w1_P1(aa_p,:); %each row = gradient for Ui (i=1 to N)

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

    w1_M1_inc = momentum*w1_M1_inc - epsilon*dw1_M1/N; 
    w1_M1 =  w1_M1 + w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc - epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 + w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
  pred_out = 1+4*arrayfun(g,sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2));
  rating=1+4*rating;
  err_train(epoch) = sqrt(sum((pred_out- rating).^2)/pairs_tr);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));

  pred_out = 1+4*arrayfun(g,sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2));
  ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
  ff = find(pred_out<1); pred_out(ff)=1;

  err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/pairs_pr);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              epoch, batch, err_train(epoch), err_valid(epoch));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if (rem(epoch,10))==0
     save pmf_weight w1_M1 w1_P1
  end

end 
end