function U=newU(U,V,c)
%input U,V such that UV' predicts R-c
%output U such that UV' predicts R
%want to find optimal row vector a st U(i,:)=U(i,:)+a
A=(V'*V)\(V');
a=c*sum(A,2);
U=U+repmat(a',length(U),1);
end
