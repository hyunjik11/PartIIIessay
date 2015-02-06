function [w,fval,exitflag]=mypmf(Au,Av,D,l_u,l_v,guess)
%function to minimise objective sum{(R-U'V)^2}+l_u*L2(U)+l_v*L2(V) wrt U and V using fminunc
%let Au,Av be the data matrix with 3cols: movie,user,rating
%note Au,Av need to be of type double
%Assume Au is ordered by user in ascending order
%Assume Av is ordered by movie in ascending order
%use parametrisation W=(U1;...;UN;V1;...;VM)
%D, the nrows of U/V
I=@(x) [x(1);x(1+find(diff(x)))]; %returns col vector with no repetitions
%[1;1+find(diff(x))] is the column vector of indices where we have a new
%element.
N=size(I(Au(:,1)),1); %N is the number of users
M=size(I(Av(:,2)),1); %M is number of movies
[~,usermapu]=ismember(Au(:,1),I(Au(:,1))); %maps user IDs in Au to set of indices 1:N
[~,usermapv]=ismember(Av(:,1),I(Au(:,1))); %maps user IDs in Av to set of indices 1:N
[~,moviemapu]=ismember(Au(:,2),I(Av(:,2)));%maps movie IDs in Au to set of indices 1:M
[~,moviemapv]=ismember(Av(:,2),I(Av(:,2)));%maps movie IDs in Av to set of indices 1:M

%input must be a scalar or a col vector
reparam=@(W) reshape(W,D,numel(W)/D);
gu=@(W) sum(W(:,usermapu).*W(:,N+moviemapu)); %length R row vector of (U_i)^T(V_j) with R=#obs
gv=@(W) sum(W(:,usermapv).*W(:,N+moviemapv)); %length R row vector of (U_i)^T(V_j) with R=#obs
%To compute gradient of sum{(R-U'V)^2} wrt W=(U,V):
Bu=@(W) bsxfun(@times,W(:,N+moviemapu),Au(:,3)'-gu(W)); %Bu is a DxR matrix 
Bv=@(W) bsxfun(@times,W(:,usermapv),Av(:,3)'-gv(W)); %Bv is a DxR matrix
%Now want: sum of first h(1) cols of B; sum of next h(2) cols of B; ...
%where h(n)=#entries with nth id/user, ie. h=count(Au(:,2)) or count(Av(:,1));
%use a cell array
Cu=@(W) mat2cell(Bu(W),D,count(Au(:,2))')';
Cv=@(W) mat2cell(Bv(W),D,count(Av(:,1))')';
C=@(W) [Cu(W);Cv(W)];
%then sum across columns of each cell of C, and convert cell into array
grad=@(W) -2*cell2mat(cellfun(@(x) sum(x,2),C(reparam(W)),'UniformOutput',false))+2*W.*[l_u*ones([N*D,1]);l_v*ones([M*D,1])];

f=@(W) (Au(:,3)'-gu(W))*(Au(:,3)-gu(W)')+l_u*sum(sum(W(:,1:N).^2))+l_v*sum(sum(W(:,N:end).^2));
F=@(W) f(reparam(W)); %applies f to W reshaped into W=(U,V);
%options = optimoptions(@fminunc,'GradObj','on','Algorithm','trust-region');
[w,fval,exitflag]=fminunc(F,guess);



