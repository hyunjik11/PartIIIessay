load results_by_freq30D_after30epochs
load results_by_freq60D
figure;
subplot(1,2,1);
hold on;
plot(pmf30freq,'ro-');
plot(rawpmf30freq,'gx-');
plot(bayespmf30freq, 'b*-');
plot(vb30freq,'cs-');
plot(rawbayespmf_vbinit30freq, 'kd-');
legend('PMF','rawPMF','BayesPMF','VB','BayesPMF\_VBinit');
xlabel('Number of observed ratings');
ylabel('RMSE');
axis([0 10 0.79 1.37]);
set(gca,'YTick', 0.75:0.05:1.40);
set(gca,'XTick', 1:9);
set(gca,'XTickL',{'1-5','6-10','11-20','21-40','41-80','81-160','161-320','321-640','>640'});
hold off;
subplot(1,2,2);
hold on;
plot(pmf60freq,'ro-');
plot(rawpmf60freq,'gx-');
plot(bayespmf60freq, 'b*-');
plot(vb60freq,'cs-');
plot(bayespmf_vbinit60freq, 'kd-');
legend('PMF','rawPMF','BayesPMF','VB','BayesPMF\_VBinit');
xlabel('Number of observed ratings');
ylabel('RMSE');
axis([0 10 0.79 1.37]);
set(gca,'YTick', 0.75:0.05:1.40);
set(gca,'XTick', 1:9);
set(gca,'XTickL',{'1-5','6-10','11-20','21-40','41-80','81-160','161-320','321-640','>640'});
hold off;
fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',9)