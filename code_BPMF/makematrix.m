% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  
function R=makematrix(dataM,num_p,num_m)
%use trainM: data ordered by movie
R = zeros(num_p,num_m); %for Netflix data, use sparse matrix instead. 
i=[0;find(diff(dataM(:,2)));length(dataM(:,2))];
for j=1:num_m
   R(dataM(i(j)+1:i(j+1),1)',j)=dataM(i(j)+1:i(j+1),3)';
   fprintf('%6i out of %6i iterations completed \n',j,num_m);
end

