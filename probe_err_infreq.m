function err=probe_err_infreq(U,V,probe_vec,mean_rating,users)
%input U,V such that UV' predicts R-mean_rating
%output is RMSE on the given 'users' in the probe set.
probe_vec=probe_vec(ismember(probe_vec(:,1),users),:);

pred_out=pred(V,U,probe_vec,mean_rating);
temp = (double(probe_vec(:,3)) - pred_out).^2;
err = sqrt(sum(temp)/length(probe_vec));
end
