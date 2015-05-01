load all_data
rating_count=zeros(5,1);
for i=1:5
    rating_count(i)=length(find(trainU(:,3)==i));
end
total=sum(rating_count);

moviecount=count(trainU(:,1));
total_users=length(moviecount);
usercount=zeros(9,1);
usercount(1)=length(find(moviecount<=5));;
for i=1:7
    usercount(i+1)=length(find(moviecount<=5*(2^i) & moviecount>5*(2^(i-1))));
end
usercount(9)=length(find(moviecount>640));

figure;
subplot(1,2,1);
bar(1:5,rating_count/total);
xlabel('Rating');
ylabel('Proportion of ratings');
subplot(1,2,2);
bar(1:9,usercount/total_users);
xlabel('Number of observed ratings');
ylabel('Proportion of users');
set(gca,'XTick', 1:9);
set(gca,'XTickL',{'1-5','6-10','11-20','21-40','41-80','81-160','161-320','321-640','>640'});
fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',9);