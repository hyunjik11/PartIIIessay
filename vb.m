%Variational Bayes algorithm for collaborative filtering on Netflix Data
%Need to write code so that I can continues to run the algorithm for more
%epochs even after it has stopped.
function [U,V,Psi,sigma,tau]=vb(train_vec,probe_vec,maxepoch,num_feat,...
    U,V,Psi,sigma,tau)
%note sigma=sigma^2, tau=tau^2
%use trainU as train_vec
%U is num_p by num_feat, V is num_m by num_feat
%when initialising U,V, note that for pmf and bayespmf U'V gives
%rating-mean_rating
%so here we also try to learn U,V st U'V gives rating-mean_rating
rand('state',0);
randn('state',0);

num_p=480189;
num_m=17770;
mean_rating = mean(train_vec(:,3));
ratings_test = double(probe_vec(:,3));
pairs_tr = length(train_vec);
pairs_pr = length(probe_vec);
index=[0;find(diff(train_vec(:,1)));pairs_tr]; %vector st ith user has the (index(i)+1)th rating to the index(i+1)th rating.

overall_err=zeros(maxepoch,1);

%%%%%%%%%%%%%%%%%%initialisation%%%%%%%%%%%%%%%%%%%%
if ~exist('U','var') 
    U=0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
end
if ~exist('V','var')
    V=0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
end
if ~exist('Psi','var')
    Psi=zeros(num_feat,num_feat,num_m); 
end
if ~exist('tau','var')
    tau=0.456;
end

%normalise each col of V st each col has L2 norm 1/sqrt(num_feat)
%then multiply each col of U by appropriate factor to keep UV' the same
norms=sqrt(num_feat*diag(V'*V));
diag_matrix_norms=diag(norms);
inv_diag_matrix_norms=diag(1./norms);
U=U*diag_matrix_norms;
V=V*inv_diag_matrix_norms;

%initialise sigma if it hasn't been given
if ~exist('sigma','var')
    sigma=diag(U'*U)/(num_p-1);
end

%%%%%%%%%%%%%%%%%%%%%%% end of initialisation %%%%%%%%%%%%%%%%%%%


for epoch=1:maxepoch
    %%%%%%%%%%%%%%%%%%%%%%% main vb algorithm  %%%%%%%%%%%%%%%%%%
    TV=zeros(num_m,num_feat);
    outerV=zeros(num_feat,num_feat,num_m); %container for Psi_j+V_j'*V_j
    for iter=1:num_m
        outerV(:,:,iter)=Psi(:,:,iter)+V(iter,:)'*V(iter,:); %only need old Psi in this expression. Then can get rid of Psi
        Psi(:,:,iter)=eye(num_feat)*num_feat; %reinitialise Psi
    end
    sigma_new=zeros(num_feat,1);
    tau_new=0;
    for i=1:num_p
        j=train_vec((index(i)+1):index(i+1),2);%set of indices N(i) ie. movies watched by user i
        Phi=inv(diag(1./sigma)+sum(outerV(:,:,j),3)/tau);
        mij=double(train_vec((index(i)+1):index(i+1),3)-mean_rating); %col vector of rating-mean_rating of user i for movies j
        TU=(sum(diag(mij)*V(j,:),1))*Phi'/tau; 
        outerU=Phi+TU'*TU; %container for Phi_j+U_j'*U_j
        sigma_new=sigma_new+diag(TU);
        for k=1:length(j)
            Psi(:,:,j(k))=Psi(:,:,j(k))+outerU/tau;
            TV(j(k),:)=TV(j(k),:)+mij(k)*TU/tau;
            tau_new=tau_new+trace(outerU*outerV(:,:,j))+mij(k)^2-2*mij(k)*TU*V(j(k),:)';
        end 
        U(i,:)=TU;
        fprintf(1,'iteration %d \n',i);
    end
    TV=TV/tau;
    for j=1:num_m
        Psi(:,:,j)=inv(Psi(:,:,j));
        V(j,:)=TV(j,:)*Psi(:,:,j)';
    end
    sigma=(sigma_new+sum(U.^2,1)')/(num_p-1);
    tau=tau_new/(pairs_tr-1);
    
    %%%%%%%%%%%%%%%%%%%%%%% end of vb algorithm  %%%%%%%%%%%%%%%%%%
    
    %%%%%%% Make predictions on the validation data %%%%%%%%%%%%%%
    probe_rat = pred(V,U,probe_vec,mean_rating);
       
    temp = (ratings_test - probe_rat).^2;
    err = sqrt(sum(temp)/pairs_pr);

    overall_err(epoch)=err;

    fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);
    
end
save /alt/applic/user-maint/hjk42/vb_random30 U V overall_err Psi sigma tau
end



