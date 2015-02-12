load all_data
tic
pmf(trainM,probeM,50,0.01,0.001,0.9,30,991,17770,480189,30);
toc
pmftime10D=toc;
save('times30.mat','pmftime10D');
tic
pmf2(trainM,probeM,50,0.01,0.001,0.9,30,991,17770,480189,30);
toc
pmf2time10D=toc;
save('times30.mat','pmf2time10D','-append');