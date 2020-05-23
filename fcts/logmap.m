function U = logmap(X,S)
% Noémie Jaquier, 2018
%
% This function computes the logarithmic map on the SPD manifold.
%
% Parameters:
%   - X:        SPD matrix
%               or SPD matrices d x d x N
%   - S:        Base SPD matrix
%
% Returns:
%   - U:        Symmetric matrix Log_S(X)
%               or symmetric matrices d x d x N

N = size(X,3);

for n = 1:N
    % 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
    % 	U(:,:,n) = S * logm(S\X(:,:,n));
    [v,d] = eig(S\X(:,:,n));
    U(:,:,n) = S * v*diag(log(diag(d)))*v^-1;
end
