%making trainM,trainU,probeM,probeU from hdf5 files
Av=hdf5read('training_by_movie.hdf5','training_by_movie'); %ordered by movie
Av=Av'; %so that we have three cols, but have movies in first col. want users instead
temp=Av(:,1);
Av(:,1)=Av(:,2);
Av(:,2)=temp; %now have users in col1, movies in col2, ratings in col3
clear temp
Au=sortrows(Av,1); %order by users
I=@(x) [x(1);x(1+find(diff(x)))]; %returns col vector with no repetitions
%[1;1+find(diff(x))] is the column vector of indices where we have a new
%element.
label_users=I(Au(:,1));
label_movies=I(Av(:,2));
N=size(I(Au(:,1)),1); %N is the number of users
M=size(I(Av(:,2)),1); %M is the number of movies
[~,usermapu]=ismember(Au(:,1),label_users); %maps user IDs in Au to set of indices 1:N
[~,usermapv]=ismember(Av(:,1),label_users); %maps user IDs in Av to set of indices 1:N
[~,moviemapu]=ismember(Au(:,2),label_movies);%maps movie IDs in Au to set of indices 1:M
[~,moviemapv]=ismember(Av(:,2),label_movies);%maps movie IDs in Av to set of indices 1:M
Au(:,1)=usermapu;
Av(:,1)=usermapv;
Au(:,2)=moviemapu;
Av(:,2)=moviemapv;
trainU=Au;
trainM=Av;
clearvars Au Av usermapu usermapv moviemapu moviemapv

Av=hdf5read('probe_by_movie.hdf5','probe_by_movie'); %ordered by movie
Av=Av'; %so that we have three cols, but have movies in first col. want users instead
temp=Av(:,1);
Av(:,1)=Av(:,2);
Av(:,2)=temp; %now have users in col1, movies in col2, ratings in col3
clear temp
Au=sortrows(Av,1); %order by users
I=@(x) [x(1);x(1+find(diff(x)))]; %returns col vector with no repetitions
%[1;1+find(diff(x))] is the column vector of indices where we have a new
%element.
PN=size(I(Au(:,1)),1); %N is the number of users
PM=size(I(Av(:,2)),1); %M is the number of movies
[~,usermapu]=ismember(Au(:,1),label_users); %maps user IDs in Au to set of indices 1:N
[~,usermapv]=ismember(Av(:,1),label_users); %maps user IDs in Av to set of indices 1:N
[~,moviemapu]=ismember(Au(:,2),label_movies);%maps movie IDs in Au to set of indices 1:M
[~,moviemapv]=ismember(Av(:,2),label_movies);%maps movie IDs in Av to set of indices 1:M
Au(:,1)=usermapu;
Av(:,1)=usermapv;
Au(:,2)=moviemapu;
Av(:,2)=moviemapv;
probeU=Au;
probeM=Av;
clearvars Au Av usermapu usermapv moviemapu moviemapv