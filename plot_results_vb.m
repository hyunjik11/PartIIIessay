load results_30D_30epochs
load results_60D_30epochs
figure;
hold on;
plot(vb30,'ro-');
plot(vb60,'b*-');
legend('VB30','VB60');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.91 0.93]);
set(gca,'YTick', 0.91:0.005:0.93);
hold off;