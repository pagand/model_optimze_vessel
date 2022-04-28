function [effective_wind] = windeffective(M14,M34,M35,se,ind)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
effective_wind_factor = cos((M14(se(ind,1):se(ind,2))-M34(se(ind,1):se(ind,2)))*pi/180);
effective_wind = M35(se(ind,1):se(ind,2)).*effective_wind_factor;
end

