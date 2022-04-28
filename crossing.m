function [se1,se2] = crossing(V1,V2,Th,Th2)

if nargin < 3
  Th=800;
  Th2 = 100;
end

indicator1 = V1>Th;
indicator2 = V2>Th;

mutual = indicator1&indicator2;

period1 = find(double(indicator1)==1);
period2 = find(double(indicator2)==1);

diff1 = double([indicator1; 0]) - double([0 ;indicator1]);
starti1 = find(double(diff1)==1);
endi1 = find(double(diff1)==-1)-1;
diff2 = double([indicator2 ;0]) - double([0; indicator2]);
starti2 = find(double(diff2)==1);
endi2 = find(double(diff2)==-1)-1;

starti1c = starti1;
starti1c(starti1<11) = [];
endi1(starti1<11) = [];
starti1 = starti1c;

% justify the results
for i=length(starti1):-1:1
    if mean(V1(starti1(i)-10:starti1(i)))>Th || ...
            mean(V1(endi1(i):endi1(i)+10))>Th || ...
                mean(V1(starti1(i):endi1(i)))<Th || ...
                    endi1(i)-  starti1(i) <Th2 
       starti1(i)=[];
        endi1(i)=[];     
    end  
end


starti2c = starti2;
starti2c(starti2<11) = [];
endi1(starti2<11) = [];
starti2 = starti2c;

for i=length(starti2):-1:1
    if mean(V2(starti2(i)-10:starti2(i)))>Th || ...
            mean(V2(endi2(i):endi2(i)+10))>Th || ...
                mean(V2(starti2(i):endi2(i)))<Th || ...
                    endi2(i)-  starti2(i) <Th2
       starti2(i)=[];
        endi2(i)=[];     
    end  
end


se1=[starti1 endi1];
se2=[starti2 endi2];

end

