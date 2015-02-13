load all_data
%tic
%pmf(trainM,probeM,50,0.01,0.001,0.9,30,991,17770,480189,10);
%toc
%pmftime10D=toc;
%save('times10.mat','pmftime10D');
tic
pmf3(trainM,probeM,50,0.01,0.001,0.9,30,991,17770,480189,10);
toc
pmf3time10D=toc;
save('times10.mat','pmf3time10D','-append');