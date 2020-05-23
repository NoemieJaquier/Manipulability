function x = expmap_vec(u,s)
% Noémie Jaquier, 2018
%
% This function computes the exponential map on the SPD manifold with 
% Mandel notation.
%
% Parameters:
%   - U:        Symmetric matrix on the tangent space of S in vector form
%               or symmetric matrices d x d x N in vector form
%   - S:        Base SPD matrix in vector form
% 
% Returns:
%   - X:        SPD matrix Exp_S(U) in vector form
%               or SPD matrices d x d x N in vector form

U = vec2symmat(u);
S = vec2symmat(s);
X = expmap(U,S);
x = symmat2vec(X);
end