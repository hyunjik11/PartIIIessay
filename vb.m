%Variational Bayes algorithm for collaborative filtering on Netflix Data
%Need to write code so that I can continues to run the algorithm for more
%epochs even after it has stopped.
function [rho,sigma,tau,U,V,Psi]=vb(train_vec,probe_vec,maxepoch,num_m,num_p,...
    num_feat,U,V,Psi)
%note sigma=sigma^2, tau=tau^2
%use trainU as train_vec
%U is num_p by num_feat, V is num_m by num_feat
rand('state',0);
randn('state',0);
if nargin<7 
    V=0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
    U=0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
    Psi=repmat(eye(num_feat)/num_feat,1,1,num_m);   
end
S=repmat(num_feat*eye(num_feat),1,1,num_m);
t=zeros(num_m,num_feat);
sigma=ones(num_feat,1);
tau=1;
rho=ones(num_feat,1)/num_feat;
sigma_new=zeros(num_feat,1);
for i=1:num_p
    j=train_vec((train_vec(:,1)==i),2);%set of indices N(i) ie. movies watched by user i
    Phi=inv(diag(1./sigma)+(sum(Psi(:,:,j),3)+V(j,:)'*V(j,:))/tau);
    U(i,:)=Phi*sum(diag(V(j,:)*U(i,:)')*V(j,:))/tau; 
    S(:,:,j)=S(:,:,j)+repmat((Phi+U(i,:)'*U(i,:))/tau,1,1,length(j));
    t(j,:)=t(j,:)+V(j,:)*U(i,:)'*U(i,:)/tau;
    sigma_new=sigma_new+(diag(Phi)+U(i,:)'.^2)/(num_p-1);
end