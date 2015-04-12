function err=probe_err(U,V,probe_vec,mean_rating)
%input U,V such that UV' predicts R-mean_rating
%output is RMSE on probe
pred_out=pred(V,U,probe_vec,mean_rating);
temp = (double(probe_vec(:,3)) - pred_out).^2;
err = sqrt(sum(temp)/length(probe_vec));
end

