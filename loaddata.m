clear;clc;close all;
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
M = readmatrix('queenCsvOut.csv',opts);
M(isnan(M))=0;


opts.SelectedVariableNames = [1:1]; 
date = readmatrix('queenCsvOut.csv',opts);

% for python
opts.SelectedVariableNames = [1:41]; 
M0 = readtable('queenCsvOut.csv',opts);
T = addvars(M0,trip,'After','WIND_SPEED_TRUE');

% 
% find mod1 for HN as 1 and NH as -1
a=zeros(1098854,1);
for i=1:length(se1)
    a(se1(i,1):se1(i,2))=1;
end
for i=1:length(se2)
    a(se2(i,1):se2(i,2))=-1;
end
Mode1 = a;
T = addvars(M0,Mode1,'After','WIND_SPEED_TRUE');
T = addvars(T,Trip,'After','Mode1');

writetable(T,'queenCsvOutAugmented.csv')  

