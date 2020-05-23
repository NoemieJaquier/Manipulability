function X = expmap(U,S)
% Noémie Jaquier, 2018
%
% This function computes the exponential map on the SPD manifold.
%
% Parameters:
%   - U:        Symmetric matrix on the tangent space of S
%               or symmetric matrices d x d x N
%   - S:        Base SPD matrix
% 
% Returns:
%   - X:        SPD matrix Exp_S(U)
%               or SPD matrices d x d x N

N = size(U,3);
for n = 1:N
% 	X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
	[v,d] = eig(S\U(:,:,n));
	X(:,:,n) = S * v*diag(exp(diag(d)))*v^-1;
end
end