close all;clc;
% Speed_1, Speed_2
V1 = M(:,26);
V2 = M(:,27);
[se1,se2] = crossing(V1,V2);

for i=1:length(se1)
    se1(i,3) = sum(abs(M(se1(i,1):se1(i,2),8)));
    
end
for i=1:length(se2)
se2(i,3) = sum(abs(M(se2(i,1):se2(i,2),13)));    
end

subplot(211);plot(se1(:,3));
subplot(212);plot(se2(:,3));
fuel_refined1 = se1(:,3);
fuel_refined2 = se2(:,3);

fuel_refined1(se1(:,3)<1e5)=1e7;
fuel_refined2(se2(:,3)<1e5)=1e7;

[m1, ind1] = min(fuel_refined1)
[m2, ind2] = min(fuel_refined2)

%inds1 = find(fuel_refined1 == m1)
%inds2 = find(fuel_refined2 == m2)

date(se1(ind1))
date(se2(ind2))


%find more  (run multiple times)
% display('more')
% fuel_refined1(ind1) = m1*10;
% fuel_refined2(ind2)= m2*10;
% [m1, ind1] = min(fuel_refined1)
% [m2, ind2] = min(fuel_refined2)
% date(se1(ind1))
% date(se2(ind2))

figure;
ind_cmp = 800
% Vessel speed (SOG)
subplot(211);plot(M(se1(ind_cmp,1):se1(ind_cmp,2),23),'r.-'); hold on;
plot(M(se1(ind1,1):se1(ind1,2),23),'b');

ylabel('SOG (H-N)')
subplot(212);plot(M(se2(ind_cmp,1):se2(ind_cmp,2),23),'r.-'); hold on;
plot(M(se2(ind2,1):se2(ind2,2),23),'b');
ylabel('SOG (N-H)')
% Time crossing 
time_cross_min1 = se1(ind1,2)-se1(ind1,1)
time_cross_min2 = se2(ind2,2)-se2(ind2,1)
time_cross_1 = se1(ind_cmp,2)-se1(ind_cmp,1)
time_cross_2 = se2(ind_cmp,2)-se2(ind_cmp,1)

% Wind speed
figure;
plot(M(se2(ind_cmp,1):se2(ind_cmp,2),35),'r.-'); hold on;
plot(M(se1(ind1,1):se1(ind1,2),35),'b'); hold on;
plot(M(se2(ind_cmp,1):se2(ind_cmp,2),23),'g'); hold on;
plot(M(se2(ind2,1):se2(ind2,2),35),'k')
legend('H-N norm', 'H-N min', 'N-H norm', 'N-H min')
ylabel('Wind')

% wind norm:
display('wind')
norm(M(se1(ind1,1):se1(ind1,2),35),1)
norm(M(se2(ind2,1):se2(ind2,2),35),1)
norm(M(se1(ind_cmp,1):se1(ind_cmp,2),35),1)
norm(M(se2(ind_cmp,1):se2(ind_cmp,2),35),1)



%deviation in heading, latiitude and longitude
% heading sin
display('heading')
var(sin(M(se1(ind1,1):se1(ind1,2),14)*pi/180))
var(sin(M(se2(ind2,1):se2(ind2,2),14)*pi/180))
var(sin(M(se1(ind_cmp,1):se1(ind_cmp,2),14)*pi/180))
var(sin(M(se2(ind_cmp,1):se2(ind_cmp,2),14)*pi/180))
% latiitude
display('lattitude')
var(M(se1(ind1,1):se1(ind1,2),15))
var(M(se2(ind2,1):se2(ind2,2),15))
var(M(se1(ind_cmp,1):se1(ind_cmp,2),15))
var(M(se2(ind_cmp,1):se2(ind_cmp,2),15))
% longitude
display('longitude')
var(M(se1(ind1,1):se1(ind1,2),16))
var(M(se2(ind2,1):se2(ind2,2),16))
var(M(se1(ind_cmp,1):se1(ind_cmp,2),16))
var(M(se2(ind_cmp,1):se2(ind_cmp,2),16))


%% plot the map
figure
scatter(M(se1(ind1,1):se1(ind1,2),16),M(se1(ind1,1):se1(ind1,2),15),'.');
hold on
scatter(M(se2(ind2,1):se2(ind2,2),16),M(se2(ind2,1):se2(ind2,2),15),'.r');
hold on
scatter(M(se1(ind_cmp,1):se1(ind_cmp,2),16),M(se1(ind_cmp,1):se1(ind_cmp,2),15),'.k');
hold on;
scatter(M(se2(ind_cmp,1):se2(ind_cmp,2),16),M(se2(ind_cmp,1):se2(ind_cmp,2),15),'.g');
legend('H-N min', 'N-H min', 'H-N norm', 'N-H norm')
ylabel('lattitude')
xlabel('longitude')
axis([-124 -123.2 49.15   49.4])