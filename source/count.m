function h=count(u) 
%function to return a column vector of the counts of each user/movie id
%where u is a column vector of the user/movie ids in ascending order
y=find(diff(u));
z=diff(y);
h=[y(1);z;size(u,1)-y(end)];
end