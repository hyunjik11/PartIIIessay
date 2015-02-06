function G=Grad(f,W)
    %computes numerical gradient of f at W, returning it as a matrix
    function A=only(i)
        %returns matrix with same dimensions as W 
        %but with 1 at the ith position and 0s everywhere else
        A=zeros(size(W));
        A(i)=1;
    end
    dW=1e-7;
    i=1:numel(W);
    G=(f(W+dW*only(i))-f(W))/dW;
end