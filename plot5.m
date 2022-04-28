close all;clc;
% SOG
V = M(:,23);
P1 = M(:,20);
P2 = M(:,21);

[se1,se2] = crossing_onlymod1(V,P1,P2);
ind_cmp = 700;

% correct the fuel consiumptiop from 8 to 7
for i=1:length(se1)
    se1(i,3) = sum(abs(M(se1(i,1):se1(i,2),7)));
    
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
    if se1(i,3)<700 || se1(i,3)>1150
        se1(i,:) = [];
    end
end

for i=size(se2,1):-1:1
    if i>size(se2,1)
        break
    end
    if se2(i,3)<800 || se2(i,3)>1200
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
% display('more')
% fuel_refined1(ind1) = m1*10;
% fuel_refined2(ind2)= m2*10;
% [m1, ind1] = min(fuel_refined1);
% [m2, ind2] = min(fuel_refined2);
% date(se1(ind1))
% date(se2(ind2))

% second method (find multiple at then same time)
y = sort(fuel_refined1,'ascend');
ind_HN1 = find(fuel_refined1 == y(1));
ind_HN2 = find(fuel_refined1 == y(2));
ind_HN3 = find(fuel_refined1 == y(3));

y = sort(fuel_refined2,'ascend');
ind_NH1 = find(fuel_refined2 == y(1));
ind_NH2 = find(fuel_refined2 == y(2));
ind_NH3 = find(fuel_refined2 == y(3));
% 
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
% fuel consumption min/max plot
MaxTripDuration_HN = max(se1(:,2)-se1(:,1));
MaxTripDuration_NH = max(se2(:,2)-se2(:,1));
FHN = M(:,7);
FNH = M(:,13);
FHN(1) = 0;FNH(1) = 0;

for i=1:MaxTripDuration_HN-1
    index = se1(:,1)+i-1;
    index(se1(:,1)+i-1>se1(:,2))=1;
    Block1(i) = mean(FHN(index));
    SD1=std(FHN(index));
    NSDR1(i)=(Block1(i)-SD1);
    PSDR1(i)=(Block1(i)+SD1);
end
for i=1:MaxTripDuration_NH-1
    index = se2(:,1)+i-1;
    index(se2(:,1)+i>se2(:,2))=1;
    Block2(i) = mean(FNH(index));
    SD2=std(FNH(index));
    NSDR2(i)=(Block2(i)-SD2);
    PSDR2(i)=(Block2(i)+SD2);
end
trip_HN = 1:MaxTripDuration_HN-1;
trip_NH = 1:MaxTripDuration_NH-1;

    
figure()
plot(trip_HN,(NSDR1),'g'); 
hold on;
plot(trip_HN,(PSDR1),'g');
patch([trip_HN fliplr(trip_HN)],[PSDR1 fliplr(NSDR1)], [0.85, 1, 0.85]);
plot(trip_HN(:), Block1, 'k--')
hold on
plot(FHN(se1(ind1,1):se1(ind1,2)),'r');hold on;
plot(FHN(se1(ind_HN2,1):se1(ind_HN2,2)),'b');hold on;
plot(FHN(se1(ind_HN3,1):se1(ind_HN3,2)),'c');
legend('\mu+\sigma','\mu-\sigma','area','\mu','min','2nd min','3rd min')
xlabel('Time (min)');ylabel('Fuel consumption H-N(Kg)')

figure()
plot(trip_NH,(NSDR2),'g'); 
hold on;
plot(trip_NH,(PSDR2),'g');
patch([trip_NH fliplr(trip_NH)],[PSDR2 fliplr(NSDR2)], [0.85, 1, 0.85]);
plot(trip_NH(:), Block2, 'k--')
hold on
plot(FNH(se2(ind2,1):se2(ind2,2)),'r');hold on;
plot(FNH(se2(ind_NH2,1):se2(ind_NH2,2)),'b');hold on;
plot(FNH(se2(ind_NH3,1):se2(ind_NH3,2)),'c');
legend('\mu+\sigma','\mu-\sigma','area','\mu','min','2nd min','3rd min')
xlabel('Time (min)');ylabel('Fuel consumption N-H(Kg)')

% fuel consumption vs speed
figure();
scatter(M(se1(ind1,1):se1(ind1,2),23),FHN(se1(ind1,1):se1(ind1,2)));hold on
scatter(M(se1(ind_HN2,1):se1(ind_HN2,2),23),FHN(se1(ind_HN2,1):se1(ind_HN2,2)),'r');hold on;
scatter(M(se1(ind_HN3,1):se1(ind_HN3,2),23),FHN(se1(ind_HN3,1):se1(ind_HN3,2)),'k');hold on;
scatter(M(se2(ind2,1):se2(ind2,2),23),FNH(se2(ind2,1):se2(ind2,2)),'b.');hold on
scatter(M(se2(ind_NH2,1):se2(ind_NH2,2),23),FNH(se2(ind_NH2,1):se2(ind_NH2,2)),'r.');hold on;
scatter(M(se2(ind_NH3,1):se2(ind_NH3,2),23),FNH(se2(ind_NH3,1):se2(ind_NH3,2)),'k.');hold on;
plot([17.5 20],[300 750]);
xlabel('SOG (knot)')
ylabel('Fuel consumption (Kg)')
legend('min H-N','2nd min H-N','3rd min H-N','min N-H','2nd min N-H','3rd min N-H')
grid


% Fuel consumption over time
figure();subplot(211);
plot(fuel_refined1)
freq = 1:250:size(se1,1);
xticks(freq)
xticklabels(date(se1(freq)))
xtickangle(45)

subplot(212);
plot(fuel_refined2)
freq = 1:250:size(se2,1);
xticks(freq)
xticklabels(date(se2(freq)))
xtickangle(45)


% bar diagram of top 3 and worst 3
y = sort(fuel_refined1,'descend');
ind_HN1w = find(fuel_refined1 == y(1));
ind_HN2w = find(fuel_refined1 == y(2));
ind_HN3w = find(fuel_refined1 == y(3));
y = sort(fuel_refined2,'descend');
ind_NH1w = find(fuel_refined2 == y(1));
ind_NH2w = find(fuel_refined2 == y(2));
ind_NH3w = find(fuel_refined2 == y(3));


fHN = fuel_refined1([ind_HN1,ind_HN2,ind_HN3,ind_HN1w,ind_HN2w,ind_HN3w]);
fNH = fuel_refined2([ind_NH1,ind_NH2,ind_NH3,ind_NH1w,ind_NH2w,ind_NH3w]);
windHN=[mean(M(se1(ind1,1):se1(ind1,2),35)),mean(M(se1(ind_HN2,1):se1(ind_HN2,2),35)),...
        mean(M(se1(ind_HN3,1):se1(ind_HN3,2),35)),mean(M(se1(ind_HN1w,1):se1(ind_HN1w,2),35)),...
        mean(M(se1(ind_HN2w,2):se1(ind_HN2w,2),35)),mean(M(se1(ind_HN3w,1):se1(ind_HN3w,2),35))];
windNH=[mean(M(se2(ind2,1):se2(ind2,2),35)),mean(M(se2(ind_NH2,1):se2(ind_NH2,2),35)),...
        mean(M(se2(ind_NH3,1):se2(ind_NH3,2),35)),mean(M(se2(ind_NH1w,1):se2(ind_NH1w,2),35)),...
        mean(M(se2(ind_NH2w,2):se2(ind_NH2w,2),35)),mean(M(se2(ind_NH3w,1):se2(ind_NH3w,2),35))];
durHN = [se1(ind1,2)-se1(ind1,1),se1(ind_HN2,2)-se1(ind_HN2,1),se1(ind_HN3,2)-se1(ind_HN3,1),...
    se1(ind_HN1w,2)-se1(ind_HN1w,1),se1(ind_HN2w,2)-se1(ind_HN2w,1),se1(ind_HN3w,2)-se1(ind_HN3w,1)];
durNH = [se2(ind2,2)-se2(ind2,1),se2(ind_NH2,2)-se2(ind_NH2,1),se2(ind_NH3,2)-se2(ind_NH3,1),...
   se2(ind_NH1w,2)-se2(ind_NH1w,1),se2(ind_NH2w,2)-se2(ind_NH2w,1),se2(ind_NH3w,2)-se2(ind_NH3w,1) ];
SOGHN = [mean(M(se1(ind1,1):se1(ind1,2),23)),mean(M(se1(ind_HN2,1):se1(ind_HN2,2),23)),...
    mean(M(se1(ind_HN3,1):se1(ind_HN3,2),23)), mean(M(se1(ind_HN1w,1):se1(ind_HN1w,2),23)),...
    mean(M(se1(ind_HN2w,1):se1(ind_HN2w,2),23)),mean(M(se1(ind_HN3w,1):se1(ind_HN3w,2),23))];
SOGNH = [mean(M(se2(ind2,1):se2(ind2,2),23)),mean(M(se2(ind_NH2,1):se2(ind_NH2,2),23)),...
    mean(M(se2(ind_NH3,1):se2(ind_NH3,2),23)), mean(M(se2(ind_NH1w,1):se2(ind_NH1w,2),23)),...
    mean(M(se2(ind_NH2w,1):se2(ind_NH2w,2),23)),mean(M(se2(ind_NH3w,1):se2(ind_NH3w,2),23))];

y = [fHN/max([fHN;fNH]) fNH/max([fHN;fNH]) windHN'/max([windHN windNH]) windNH'/max([windHN windNH])...
    durHN'/max([durHN durNH]) durNH'/max([durHN durNH]) SOGHN'/max([SOGHN SOGNH]) SOGNH'/max([SOGHN SOGNH])];

figure;
bar(y);
legend('fuel H-N','Fuel N-H','Wind H-N','Wind N-H','Duration H-N','Duration N-H','SOG H-N','SOG N-H')
xticklabels({'best','2nd best','3rd best','worst','2nd worst','3rd worst'})



% Plot of top10 vs worst 10 compare speed1,speed2
y1 = sort(fuel_refined1,'ascend');
y2 = sort(fuel_refined2,'ascend');
y3 = sort(fuel_refined1,'descend');
y4 = sort(fuel_refined2,'descend');
for i=1:10
    indHN_best(i) = find(fuel_refined1 == y1(i));
    indNH_best(i) = find(fuel_refined2 == y2(i));
    indHN_worst(i) = find(fuel_refined1 == y3(i));
    indNH_worst(i) = find(fuel_refined2 == y4(i));
end

V1 = M(:,26);
V2 = M(:,27);
V1(1) = 0;
V2(1) = 0;
for i=1:MaxTripDuration_HN-1
    index1 = se1(indHN_best,1)+i-1;
    index2 = se1(indHN_worst,1)+i-1;
    %index1(se1(indHN_best,1)+i-1>se1(indHN_best,2))=1;
    %index2(se1(indHN_worst,1)+i-1>se1(indHN_worst,2))=1;  
    Block11(i) = mean(V1(index1));
    Block12(i) = mean(V1(index2));
    
    SD11=std(V1(index1));
    SD12=std(V1(index2)); 
    
    NSDR11(i)=(Block11(i)-SD11);
    NSDR12(i)=(Block12(i)-SD12);
    
    PSDR11(i)=(Block11(i)+SD11);
    PSDR12(i)=(Block12(i)+SD12);
end

figure()
plot(trip_HN,(NSDR11),'g'); hold on;
plot(trip_HN,(PSDR11),'g'); hold on;
patch([trip_HN fliplr(trip_HN)],[PSDR11 fliplr(NSDR11)], [0.85, 1, 0.85]);
plot(V1(se1(ind1,1):se1(ind1,2)),'r','LineWidth',1.5);hold on;

plot(trip_HN,(NSDR12),'y'); hold on;
plot(trip_HN,(PSDR12),'y'); hold on;
patch([trip_HN fliplr(trip_HN)],[PSDR12 fliplr(NSDR12)], 'y'); hold on
plot(V1(se1(ind_HN1w,1):se1(ind_HN1w,2)),'k','LineWidth',1.5);hold on;
legend('top_l','top_u','area top10','best','down_l','down_u','area worst10','worst')
xlabel('Time (min)');ylabel('Speed H-N(Knot)')

for i=1:MaxTripDuration_NH-1
    index1 = se2(indNH_best,1)+i-1;
    index2 = se2(indNH_worst,1)+i-1;
    %index1(se1(indHN_best,1)+i-1>se1(indHN_best,2))=1;
    %index2(se1(indHN_worst,1)+i-1>se1(indHN_worst,2))=1;  
    Block11(i) = mean(V2(index1));
    Block12(i) = mean(V2(index2));
    
    SD11=std(V2(index1));
    SD12=std(V2(index2)); 
    
    NSDR11(i)=(Block11(i)-SD11);
    NSDR12(i)=(Block12(i)-SD12);
    
    PSDR11(i)=(Block11(i)+SD11);
    PSDR12(i)=(Block12(i)+SD12);
end
figure()
plot(trip_HN,(NSDR11),'g'); hold on;
plot(trip_HN,(PSDR11),'g'); hold on;
patch([trip_HN fliplr(trip_HN)],[PSDR11 fliplr(NSDR11)], [0.85, 1, 0.85]);
plot(V2(se2(ind2,1):se2(ind2,2)),'r','LineWidth',1.5);hold on;

plot(trip_HN,(NSDR12),'y'); hold on;
plot(trip_HN,(PSDR12),'y'); hold on;
patch([trip_HN fliplr(trip_HN)],[PSDR12 fliplr(NSDR12)], 'y'); hold on
plot(V2(se2(ind_NH1w,1):se2(ind_NH1w,2)),'k','LineWidth',1.5);hold on;
legend('top_l','top_u','area top10','best','down_l','down_u','area worst10','worst')
xlabel('Time (min)');ylabel('Speed N-H(Knot)')








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




