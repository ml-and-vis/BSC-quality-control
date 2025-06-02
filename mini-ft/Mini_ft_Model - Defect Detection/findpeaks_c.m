%-------------------------------------------------------------------------%
% The purpose of this function is to find the response and loads of a time
% history response, and associated hammer response. That is for generating
% the individual subsequences used for the analysis.

function [response, hammer_hits] = findpeaks_c(I_response, I_hammer, Tlen)

% Get the peak locations of the response
[pks,locs] = findpeaks(I_response, 'MinPeakDistance', 1000);

% Get each individual hammer and response
for i= 1:length(locs)
    lower_lim = locs(i) - 20;
    upper_lim = locs(i) + 150;
        
    if (upper_lim < Tlen) && (lower_lim >= 1)
        if (max(I_response(lower_lim: upper_lim)) > 0.1) 
            response(i,:) = I_response(lower_lim: upper_lim);
            hammer_hits(i,:) = I_hammer(lower_lim: upper_lim);
        end
    end

end