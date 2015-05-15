function a=test(a,b)

if ~exist('b','var') 
    b=10;
end
a=a+b;
end