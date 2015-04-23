load results_30D_30epochs
load results_60D_30epochs
%need to update vb60 of latter so that it has 30 epochs
figure;
subplot(1,2,1);
hold on;
plot(pmf30,'ro-');
plot(rawpmf30,'gx-');
plot(bayespmf30, 'b*-');
plot(vb30,'cs-');
plot(rawbayespmf_vbinit30, 'kd-');
plot([1 30],[0.9474 0.9474], '--');
legend('PMF','rawPMF','BayesPMF','VB','BayesPMF\_VBinit','Cinematch');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.908 1.05]);
set(gca,'YTick', 0.90:0.01:1.05);
hold off;
subplot(1,2,2);
hold on;
plot(pmf60,'ro-');
plot(rawpmf60,'gx-');
plot(bayespmf60, 'b*-');
plot(vb60,'cs-');
plot(rawbayespmf_vbinit60, 'kd-');
plot([1 30],[0.9474 0.9474], '--');
legend('PMF','rawPMF','BayesPMF','VB','BayesPMF\_VBinit','Cinematch');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.908 1.05]);
set(gca,'YTick', 0.90:0.01:1.05);
hold off;