function [fval,param]=gd(f,grad,lr,p,iparam,n)
%function to perform n iterations of gradient descent
%with learning rate lr, momentum p, initial value of parameters iparam
%returns value of fn and parameters
prevparam=zeros(size(iparam));
param=iparam;
for i=1:n
    paramtemp=param;
    param=param-lr*grad(param)+p*(param-prevparam);
    prevparam=paramtemp;
end
fval=f(param);
end

