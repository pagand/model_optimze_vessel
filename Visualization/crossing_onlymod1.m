function [se1,se2] = crossing_onlymod1(V,P1,P2)


Th=16;
Th2 = 85;


indicator = V>Th;

diff = double([indicator; 0]) - double([0 ;indicator]);
starti = find(double(diff)==1);
endi = find(double(diff)==-1)-1;

startic = starti;
startic(starti<11) = [];
endi(starti<11) = [];
starti = startic;

% justify the results
j = 1;
q = 1;
for i=length(starti):-1:1
    if mean(V(starti(i)-10:starti(i)))>Th || ...
            mean(V(endi(i):endi(i)+10))>Th || ...
                mean(V(starti(i):endi(i)))<Th || ...
                    endi(i)-  starti(i) <Th2 
       starti(i)=[];
        endi(i)=[];     
    else
        p11 = mean(P1(starti(i):endi(i)));
        p22 = mean(P2(starti(i):endi(i)));
        if p11>p22
            starti1(j) = starti(i);
            endi1(j) = endi(i);
            j= j+1;
        else
            starti2(q) = starti(i);
            endi2(q) = endi(i);
            q = q+1;
        end
    end
end




    

se1=[starti1' endi1'];
se2=[starti2' endi2'];
se1 = flip(se1,1);
se2 = flip(se2,1);

end

