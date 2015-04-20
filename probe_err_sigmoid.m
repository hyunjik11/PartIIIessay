function err=probe_err_sigmoid(U,V,probe_vec)
%input U,V such that sigmoid(UV') predicts R-1/4
%output is RMSE on probe_vec
g=@(x) 1/(1+exp(-x));
  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));

  pred_out = 1+4*arrayfun(g,sum(V(aa_m,:).*U(aa_p,:),2));
  ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
  ff = find(pred_out<1); pred_out(ff)=1;

temp = (rating - pred_out).^2;
err = sqrt(sum(temp)/length(probe_vec));
end