function full=svd_em(R,n,maxIter)
%R is the matrix of observed entries
%n is the rank of SVD approximation
full=R;
for iter=1:maxIter
   [U,S,V]=svds(full);
   clear full
    D=diag(S);
    clear S
    if n>length(D)
        error('n greater than number of singular values');
    end
    Dapprox=D(1:n); %n approx
    clear D
    S=diag(Dapprox);
    clear Dapprox
    U=U(:,1:n);
    V=V(:,1:n);
    approx=U*S*(V');
    clearvars U S V
    approx(R~=0)=0;
    full=R+approx;
    clear approx
end