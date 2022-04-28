close all;clc;
% SOG
V = M(:,23);
P1 = M(:,20);
P2 = M(:,21);

[se1,se2] = crossing_onlymod1(V,P1,P2);
ind_cmp = 700;

for i=1:length(se1)
    se1(i,3) = sum(abs(M(se1(i,1):se1(i,2),8)));
    
end
for i=1:length(se2)
se2(i,3) = sum(abs(M(se2(i,1):se2(i,2),13)));    
end

se1(:,3) = se1(:,3) /60;
se2(:,3) = se2(:,3) /60;

% post process: get rid of very small
for i=size(se1,1):-1:1
    if i>size(se1,1)
        break
    end
    if se1(i,3)<1e4 /60
        se1(i,:) = [];
    end
end

for i=size(se2,1):-1:1
    if i>size(se2,1)
        break
    end
    if se2(i,3)<1e4 /60
        se2(i,:) = [];
    end
end



hist(se1(:,3),20)
ylabel('# samples')
xlabel('Fuel consumption H-N (kg)')
figure;
hist(se2(:,3),20)
ylabel('# samples')
xlabel('Fuel consumption N-H (kg)')


% post process: get rid of very small/large values
for i=size(se1,1):-1:1
    if i>size(se1,1)
        break
    end
    if se1(i,3)<1000 || se1(i,3)>1400
        se1(i,:) = [];
    end
end

for i=size(se2,1):-1:1
    if i>size(se2,1)
        break
    end
    if se2(i,3)<900 || se2(i,3)>1250
        se2(i,:) = [];
    end
end



fuel_refined1 = se1(:,3);
fuel_refined2 = se2(:,3);



[m1, ind1] = min(fuel_refined1);
[m2, ind2] = min(fuel_refined2);

%inds1 = find(fuel_refined1 == m1)
%inds2 = find(fuel_refined2 == m2)
disp('date of best practice')
date(se1(ind1))
date(se2(ind2))


%find more  (run multiple times)
display('more')
fuel_refined1(ind1) = m1*10;
fuel_refined2(ind2)= m2*10;
[m1, ind1] = min(fuel_refined1);
[m2, ind2] = min(fuel_refined2);
date(se1(ind1))
date(se2(ind2))


% Fuel
disp('fuel consumption')
f_best_HN = m1
f_best_NH = m2
f_mean_HN = mean(fuel_refined1)
f_mean_NH = mean(fuel_refined2)



disp('Vessel speed (SOG)')
SOG_best_HN = mean(M(se1(ind1,1):se1(ind1,2),23))
SOG_best_NH = mean(M(se2(ind2,1):se2(ind2,2),23))
M_mean = 0;
for i=1:size(se1,1)
    M_mean =M_mean+ mean(M(se1(i,1):se1(i,2),23));
end
SOG_mean_HN = M_mean/size(se1,1)

M_mean = 0;
for i=1:size(se2,1)
    M_mean =M_mean+ mean(M(se2(i,1):se2(i,2),23));
end
SOG_mean_NH = M_mean/size(se2,1)


disp('Vessel speed over water (STW)')
STW_best_HN = mean(M(se1(ind1,1):se1(ind1,2),28))
STW_best_NH = mean(M(se2(ind2,1):se2(ind2,2),28))
M_mean = 0;
for i=1:size(se1,1)
    M_mean =M_mean+ mean(M(se1(i,1):se1(i,2),28));
end
STW_mean_HN = M_mean/size(se1,1)

M_mean = 0;
for i=1:size(se2,1)
    M_mean =M_mean+ mean(M(se2(i,1):se2(i,2),28));
end
STW_mean_NH = M_mean/size(se2,1)


disp('time crossing')

time_best_HN = se1(ind1,2)-se1(ind1,1)
time_best_NH = se2(ind2,2)-se2(ind2,1)
time_mean_HN = mean(se1(:,2)-se1(:,1))
time_mean_NH = mean(se2(:,2)-se2(:,1))



disp('wind apparent speed')
effective_wind_factor = cos((M(se1(ind1,1):se1(ind1,2),14)-M(se1(ind1,1):se1(ind1,2),34))*pi/180);
effective_wind_best_HN = M(se1(ind1,1):se1(ind1,2),35).*effective_wind_factor;
effective_wind_best_HN_mean = mean(effective_wind_best_HN)


effective_wind_factor = cos((M(se2(ind2,1):se2(ind2,2),14)-M(se2(ind2,1):se2(ind2,2),34))*pi/180);
effective_wind_best_NH = M(se2(ind2,1):se2(ind2,2),35).*effective_wind_factor;
effective_wind_best_NH_mean = mean(effective_wind_best_NH)

effective_wind_mean_HN_mean = 0;
for i=1:size(se1,1)
effective_wind_factor = cos((M(se1(i,1):se1(i,2),14)-M(se1(i,1):se1(i,2),34))*pi/180);
effective_wind_mean_HN_mean =effective_wind_mean_HN_mean+ mean(M(se1(i,1):se1(i,2),35).*effective_wind_factor);
end
effective_wind_mean_HN_mean = effective_wind_mean_HN_mean/size(se1,1)

effective_wind_mean_NH_mean = 0;
for i=1:size(se2,1)
effective_wind_factor = cos((M(se2(i,1):se2(i,2),14)-M(se2(i,1):se2(i,2),34))*pi/180);
effective_wind_mean_NH_mean =effective_wind_mean_NH_mean+ mean( M(se2(i,1):se2(i,2),35).*effective_wind_factor);
end
effective_wind_mean_NH_mean = effective_wind_mean_NH_mean/size(se1,1)



effective_wind_factor = cos((M(se1(ind_cmp,1):se1(ind_cmp,2),14)-M(se1(ind_cmp,1):se1(ind_cmp,2),34))*pi/180);
effective_wind_random_HN = M(se1(ind_cmp,1):se1(ind_cmp,2),35).*effective_wind_factor;

effective_wind_factor = cos((M(se2(ind_cmp,1):se2(ind_cmp,2),14)-M(se2(ind_cmp,1):se2(ind_cmp,2),34))*pi/180);
effective_wind_random_NH = M(se2(ind_cmp,1):se2(ind_cmp,2),35).*effective_wind_factor;



figure;
plot(effective_wind_best_HN,'r'); hold on;
plot(effective_wind_best_NH,'b'); hold on;
plot(effective_wind_random_HN,'g'); hold on;
plot(effective_wind_random_NH,'k')
legend('H-N best', 'N-H best', 'H-N random', 'N-H random')
ylabel('Effective wind speed (knots)')
xlabel('Time (min)')
grid



%deviation in heading, latiitude and longitude
V1x = mode([M(se2(:,1),16);M(se1(:,2),16) ]);
V1y = mode([M(se2(:,1),15);M(se1(:,2),15) ]);
V2x = mode([M(se2(:,2),16);M(se1(:,1),16) ]);
V2y = mode([M(se2(:,2),15);M(se1(:,1),15) ]);

V1 = [V1x,V1y,0];
V2 = [V2x,V2y,0];

% best HN
dev = 0;
        for j = se1(ind1,1):se1(ind1,2)
            ptx = M(j,16);
            pty = M(j,15);
            pt = [ptx,pty,0];
            dev = dev + point_to_line(pt,V1,V2);
        end
dev_best_HN = dev

% besr NH
dev = 0;

        for j = se2(ind2,1):se2(ind2,2)
            ptx = M(j,16);
            pty = M(j,15);
            pt = [ptx,pty,0];
            dev = dev + point_to_line(pt,V1,V2);
        end
dev_best_NH = dev

% average HN
dev = 0;
count = 0;
for i=1:size(se1,1)
    if (M(se1(i,1):se1(i,2),15)> 49.15)
        if (M(se1(i,1):se1(i,2),15)<49.4  )
        count = count +1;
        for j = se1(i,1):se1(i,2)
            ptx = M(j,16);
            pty = M(j,15);
            pt = [ptx,pty,0];
            dev = dev + point_to_line(pt,V1,V2);
        end
        end
    end
end
dev_HN = dev/count

% average NH
dev = 0;
count = 0;
for i=1:size(se2,1)
    if (M(se2(i,1):se2(i,2),15)> 49.15)
        if(M(se2(i,1):se2(i,2),15)<49.4  )
        count = count +1;
        for j = se2(i,1):se2(i,2)
            ptx = M(j,16);
            pty = M(j,15);
            pt = [ptx,pty,0];
            dev = dev + point_to_line(pt,V1,V2);
        end
        end
    end
end
dev_NH = dev/count



%% plots

% Vessel speed (SOG)
figure;
subplot(211);
plot(M(se1(ind1,1):se1(ind1,2),23),'r');hold on;
plot(M(se1(ind_cmp,1):se1(ind_cmp,2),23),'g'); 



subplot(212);plot(M(se2(ind2,1):se2(ind2,2),23),'b');hold on;
plot(M(se2(ind_cmp,1):se2(ind_cmp,2),23),'k'); 



% Vessel speed (STW)

subplot(211); hold on;
plot(M(se1(ind1,1):se1(ind1,2),28),'-.r');hold on;
plot(M(se1(ind_cmp,1):se1(ind_cmp,2),28),'-.g'); 
legend('H-N best SOG', 'H-N random SOG','H-N best STW', 'H-N random STW')
ylabel('SOG/STW H-N (knot)')
xlabel('Time (min)')

subplot(212); hold on;plot(M(se2(ind2,1):se2(ind2,2),28),'-.b');hold on;
plot(M(se2(ind_cmp,1):se2(ind_cmp,2),28),'-.k'); 
legend('N-H best SOG', 'N-H random SOG','N-H best STW', 'N-H random STW')
ylabel('SOG/STW N-H (knot)')
xlabel('Time (min)')

% plot the map
figure
scatter(M(se1(ind1,1):se1(ind1,2),16),M(se1(ind1,1):se1(ind1,2),15),'.r');
hold on
scatter(M(se2(ind2,1):se2(ind2,2),16),M(se2(ind2,1):se2(ind2,2),15),'.b');
hold on
scatter(M(se1(ind_cmp,1):se1(ind_cmp,2),16),M(se1(ind_cmp,1):se1(ind_cmp,2),15),'.g');
hold on;
scatter(M(se2(ind_cmp,1):se2(ind_cmp,2),16),M(se2(ind_cmp,1):se2(ind_cmp,2),15),'.k');
legend('H-N best', 'N-H best', 'H-N random', 'N-H random')
ylabel('lattitude (deg)')
xlabel('longitude (deg)')
axis([-124 -123.2 49.15   49.4])
grid

% Vessel power
figure;
subplot(211);
plot(M(se1(ind1,1):se1(ind1,2),20),'r');hold on;
plot(M(se1(ind_cmp,1):se1(ind_cmp,2),20),'g'); 
legend('H-N best', 'H-N random')
ylabel('Power H-N (W)')
xlabel('Time (min)')

subplot(212);plot(M(se2(ind2,1):se2(ind2,2),21),'b');hold on;
plot(M(se2(ind_cmp,1):se2(ind_cmp,2),21),'k'); 
legend('N-H best', 'N-H random')
ylabel('Power (W)')
xlabel('Time (min)')


