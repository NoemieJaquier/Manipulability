function M = spdMean(setS, nbIt)
% Noémie Jaquier, 2018
%
% This function computes the mean of SPD matrices on the SPD manifold.
%
% Parameters:
%   - setS:     Set of SPD matrices d x d x N
%   - nbIt:     Number of iterations for the Gauss-Newton algorithm
%               (10 is typically enough)
% 
% Returns:
%   - M:        Mean SPD matrix

if nargin == 1
	nbIt = 10;
end
M = setS(:,:,1);

for i=1:nbIt
	L = zeros(size(setS,1),size(setS,2));
	for n = 1:size(setS,3)
		L = L + logm(M^-.5 * setS(:,:,n)* M^-.5);
	end
	M = M^.5 * expm(L./size(setS,3)) * M^.5;
end

end