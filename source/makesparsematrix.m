function R=makesparsematrix(dataM,num_p,num_m)
%use trainM: data ordered by movie
R = sparse(num_p,num_m); %for Netflix data, use sparse matrix instead. 
i=[0;find(diff(dataM(:,2)));length(dataM(:,2))];
for j=1:num_m
   R(dataM(i(j)+1:i(j+1),1)',j)=dataM(i(j)+1:i(j+1),3)';
   fprintf('%6i out of %6i iterations completed \n',j,num_m);
end
