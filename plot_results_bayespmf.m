load results_bayespmf30D
load results_bayespmf60D
figure;
subplot(1,2,1);
hold on;
plot(bayespmf30D_4,'ro-');
plot(bayespmf30D_8,'gx-');
plot(bayespmf30D_16, 'b*-');
plot(bayespmf30D_32, 'm+-');
legend('4 samples','8 samples','16 samples','32 samples');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.905 0.93]);
set(gca,'YTick', 0.905:0.005:0.93);
hold off;
subplot(1,2,2);
hold on;
plot(bayespmf60D_4,'ro-');
plot(bayespmf60D_8,'gx-');
plot(bayespmf60D_16, 'b*-');
plot(bayespmf60D_32, 'm+-');
legend('4 samples','8 samples','16 samples','32 samples');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.905 0.93]);
set(gca,'YTick', 0.905:0.005:0.93);
hold off;