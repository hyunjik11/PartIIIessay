function full=svd_em(R,n,maxIter)
%R is the matrix of observed entries
%n is the rank of SVD approximation
full=R;
W=R~=0; %W is the matrix with 1's at non-zeros values of R and 0's everywhere else
for iter=1:maxIter
   [U,S,V]=svd(full);
    D=diag(S);
    if n>length(D)
        error('n greater than number of singular values');
    end
    Dapprox=D(1:n); %n approx
    S=diag(Dapprox);
    U=U(:,1:n);
    V=V(1:n,:); 
    full=W.*R+(1-W).*(U*S*V);
end