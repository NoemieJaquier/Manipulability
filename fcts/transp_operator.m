function Ac = transp_operator(S1,S2)
% Noémie Jaquier, 2018
%
% This function computes the parallel transport operator from S1 to S2
% on the SPD manifold.
% A SPD matrix X is transported from S1 to S2 with Ac * X * Ac'.
%
% Parameters:
%   - S1:       SPD matrix
%   - S2:       SPD matrix
% 
% Returns:
%   - Ac:       Parallel transport operator

Ac = (S2/S1)^.5;
end