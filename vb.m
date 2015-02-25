%Variational Bayes algorithm for collaborative filtering on Netflix Data
%Need to write code so that I can continues to run the algorithm for more
%epochs even after it has stopped.
function [rho,sigma,tau,U,V,Psi,S,t]=vb(train_vec,probe_vec,maxepoch,num_m,num_p,...
    num_feat,U,V,Psi)
%note sigma=sigma^2, tau=tau^2
%use trainU as train_vec
rand('state',0);
randn('state',0);
if nargin<7 
    V=0.1*randn(num_m, num_feat); % Movie feature vectors (each row corresponds to a movie)
    U=0.1*randn(num_p, num_feat); % User feature vecators (each row corresponds to a user)
    Psi=cell(num_m,1);
    for i=1:num_m
        Psi{i}=eye(num_feat)/num_feat;
    end
    t=zeros(num_m,1);
end
S=cell(num_m,1);
for i=1:num_m
    S{i}=num_feat*eye(num_feat);
end
t=zeros(num_m,1);
sigma=ones(num_feat,1);
tau=1;
rho=ones(num_feat,1)/num_feat;

for i=1:num_p
    j=train_vec((train_vec(:,1)==i),2);%set of indices N(i) ie. movies watched by user i
    Phi=inv(diag(1./sigma)+(sum(cat(3,Psi{j}),3)+V'*V)/tau);
    %cat(3,Psi{j}) is the set of cells Psi{j} 
end