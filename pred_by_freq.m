function err=pred_by_freq(U,V,trainU,probeU,mean_rating,probe_rat_all)
%function which gives vector of length 9
%showing the RMSE of input U,V for users who have watched:
%1-5,6-10,11-20,21-40,41-80,81-160,161-320,321-640,>640 movies
moviecount=count(trainU(:,1));
err=zeros(9,1);
user=cell(9,1);
user{1}=find(moviecount<=5);
for i=1:7
    user{i+1}=find(moviecount<=5*(2^i) & moviecount>5*(2^(i-1)));
end
user{9}=find(moviecount>640);
if ~exist('probe_rat_all','var') 
    for i=1:9
        err(i)=probe_err_infreq(U,V,probeU,mean_rating,user{i});
    end
else
    for i=1:9
        indices=ismember(probeU(:,1),user{i});
        data=probeU(indices,3);
        pred=probe_rat_all(indices);
        temp=(double(data)-pred).^2;
        err(i)=sqrt(sum(temp)/length(pred));
    end
end

end


