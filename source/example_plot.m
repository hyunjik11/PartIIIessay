figure;
hold on;
plot(bayespmf60D_4,'ro-');
plot(bayespmf60D_8,'gx-');
plot(bayespmf60D_16, 'b*-');
plot(bayespmf60D_32, 'm+-');
%plot(vb30,'cs-');
%plot(rawbayespmf_vbinit30, 'kd-');
%plot([1 30],[0.9474 0.9474], '--');
legend('4 samples','8 samples','16 samples','32 samples');
xlabel('Epochs');
ylabel('RMSE');
axis([0 31 0.90 0.93]);
set(gca,'YTick', 0.90:0.005:0.93);
hold off;
%figure;
%subplot(2,2,1); autocorr(djokovic);
%subplot(2,2,2); autocorr(nadal);
%subplot(2,2,3); autocorr(fish);
%subplot(2,2,4); autocorr(elgin);