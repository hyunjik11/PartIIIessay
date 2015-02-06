load all_data
tic
[U,V]=pmf(trainU,probeU,0.005,0.01,0.001,0.9,30,991,17770,480189,10);
toc
pmftime10D=toc;
save('times.mat','pmftime10D');
tic
[U,V]=pmf2(trainU,probeU,0.005,0.01,0.001,0.9,30,991,17770,480189,10);
toc
pmf2time10D=toc;
save('times.mat','pmf2time10D','-append');