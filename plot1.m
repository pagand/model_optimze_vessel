%clear;clc;close all;
opts = detectImportOptions('queenCsvOut.csv');
%%opts.ExtraColumnsRule = 'ignore';
disp(opts)

preview('queenCsvOut.csv',opts)


opts.SelectedVariableNames = [1:1]; 
%opts.DataRange = '2:11';
%M1 = readmatrix('queenCsvOut.csv',opts);
%[day,time] = unzip_dati(M1);


opts.SelectedVariableNames = [5:41]; 
%%opts.DataRange = '2:11';
%M = readmatrix('queenCsvOut.csv',opts);
%M(isnan(M))=0;


% for i=1:1440:length(M)-1440
%     fuel((i-1)/1440+1) = norm(M(i:i+1439,16)) + norm(M(i:i+1439,17));
% end

rest_ind = find(time==0);
for i=1:length(rest_ind)-1   
    
    fuel(i) = norm(M(rest_ind(i):rest_ind(i+1)-1,16)) + norm(M(rest_ind(i):rest_ind(i+1)-1,17));
end

plot(fuel)
fuel_refined = fuel;
fuel_refined(fuel<4000)=10000;

[m, ind] = min(fuel_refined)



% ind_good = 1440*(ind-1)+1;
%period = ind_good:ind_good+1439;


%%
figure();

period = rest_ind(ind):rest_ind(ind+1)-1;
subplot(221)
plot(time(period),M(period,2));hold all;
ylabel('ENGINE 1  FLOWRATE')
subplot(222)
plot(time(period),M(period,8));hold all;
ylabel('ENGINE 1  FUEL CONSUMPTION')
subplot(223)
plot(time(period),M(period,7));hold all;
ylabel('ENGINE 2 FLOWRATE')
subplot(224)
plot(time(period),M(period,13));hold all;
ylabel('ENGINE 2 FUEL CONSUMPTION')

ind1 = 10;
period = rest_ind(ind1):rest_ind(ind1+1)-1;
subplot(221)
plot(time(period),M(period,2),'r--')
legend(day{rest_ind(ind)},day{rest_ind(ind1)})
subplot(222)
plot(time(period),M(period,8),'r--')
subplot(223)
plot(time(period),M(period,7),'r--')
subplot(224)
plot(time(period),M(period,13),'r--')




%%
figure();
period = rest_ind(ind):rest_ind(ind+1)-1;
subplot(221)
plot(time(period),M(period,26));hold all;
ylabel('SPEED 1')
subplot(222)
plot(time(period),M(period,20));hold all;
ylabel('POWER 1')
subplot(223)
plot(time(period),M(period,27));hold all;
ylabel('SPEED 2')
subplot(224)
plot(time(period),M(period,21));hold all;
ylabel('POWER 2')


ind1 = 10;
period = rest_ind(ind1):rest_ind(ind1+1)-1;
subplot(221)
plot(time(period),M(period,26),'r--')
legend(day{rest_ind(ind)},day{rest_ind(ind1)})
subplot(222)
plot(time(period),M(period,20),'r--')
subplot(223)
plot(time(period),M(period,27),'r--')
subplot(224)
plot(time(period),M(period,21),'r--')

%% Depth
figure;
period = rest_ind(ind):rest_ind(ind+1)-1;
plot(time(period),M(period,1));hold all;
ylabel('Depth')

period = rest_ind(ind1):rest_ind(ind1+1)-1;
plot(time(period),M(period,1),'r--')
legend(day{rest_ind(ind)},day{rest_ind(ind1)})


%% Wind


figure();
period = rest_ind(ind):rest_ind(ind+1)-1;
subplot(211)
plot(time(period),M(period,36));hold all;
ylabel('WIND ANGLE TRUE')
subplot(212)
plot(time(period),M(period,37));hold all;
ylabel('WIND SPEED TRUE')



ind1 = 10;
period = rest_ind(ind1):rest_ind(ind1+1)-1;
subplot(211)
plot(time(period),M(period,36),'r--')
legend(day{rest_ind(ind)},day{rest_ind(ind1)})
subplot(212)
plot(time(period),M(period,37),'r--')



