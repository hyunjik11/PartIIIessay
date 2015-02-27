%Variational Bayes algorithm for collaborative filtering on Netflix Data
%Need to write code so that I can continues to run the algorithm for more
%epochs even after it has stopped.
function [rho,sigma,tau,U,V,Psi]=vb(train_vec,probe_vec,maxepoch,num_m,num_p,...
    num_feat,U,V,Psi)
%note sigma=sigma^2, tau=tau^2
%use trainU as train_vec
%U is num_p by num_feat, V is num_m by num_feat
%when initialising U,V, note that for pmf and bayespmf U'V gives
%rating-mean_rating
%so here we also try to learn U,V st U'V gives rating-mean_rating
rand('state',0);
randn('state',0);

mean_rating = mean(train_vec(:,3));
ratings_test = double(probe_vec(:,3));
pairs_tr = length(train_vec);
pairs_pr = length(probe_vec);

overall_err=zeros(maxepoch,1);

if nargin<7 
    V=0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
    U=0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
    Psi=repmat(eye(num_feat)/num_feat,1,1,num_m);   
end

sigma=ones(num_feat,1);
tau=1;
rho=ones(num_feat,1)/num_feat;
for epoch=1:maxepoch
    S=repmat(num_feat*eye(num_feat),1,1,num_m);
    t=zeros(num_m,num_feat); %each row vector is t_j
    sigma_new=zeros(num_feat,1);
    tau_new=0;
    for i=1:num_p
        j=train_vec((train_vec(:,1)==i),2);%set of indices N(i) ie. movies watched by user i
        Phi=inv(diag(1./sigma)+(sum(Psi(:,:,j),3)+V(j,:)'*V(j,:))/tau);
        m_ij=train_vec((train_vec(:,1)==i),3)-mean_rating; %col vector of rating-mean_rating of user i for movies j
        U(i,:)=Phi*sum(diag(m_ij)*V(j,:))/tau; 
        S(:,:,j)=S(:,:,j)+repmat((Phi+U(i,:)'*U(i,:))/tau,1,1,length(j));
        t(j,:)=t(j,:)+m_ij*U(i,:)/tau;
        sigma_new=sigma_new+diag(Phi)+U(i,:)'.^2;
        %----------optional-----%
        VVt=zeros(num_feat,num_feat,length(j)); %v_jv_j' in eqn 24
        for k=1:length(j)
            VVt(:,:,k)=V(j(k),:)'*V(j(k),:);
        end
        tau_new=tau_new+sum(m_ij.^2)-2*sum(m_ij.*(V(j,:)*U(i,:)'))+...
            sum(sum(sum(repmat((Phi+U(i,:)'*U(i,:)),1,1,length(j)).*(Psi+VVt))));
        %not sure if this line should be included. 
        %Problem as we don't yet know new V
        %--------------------%
    end
    sigma=sigma_new/(num_p-1);
    tau=tau_new/(pairs_tr-1);
    for j=1:num_m
        Psi(:,:,j)=inv(S(:,:,j));
        V(j,:)=t(j,:)*(Psi(:,:,j)');
    end
    
%%%%%%% Make predictions on the validation data %%%%%%%
    probe_rat = pred(V,U,probe_vec,mean_rating);
       
    temp = (ratings_test - probe_rat).^2;
    err = sqrt(sum(temp)/pairs_pr);

    overall_err(epoch)=err;

    fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);
    
end
save /alt/applic/user-maint/hjk42/vb_weights_and_errors V U overall_err
end



