function u = logmap_vec(x,s)
% Noémie Jaquier, 2018
%
% This function computes the logarithmic map on the SPD manifold with
% Mandel notation.
%
% Parameters:
%   - X:        SPD matrix in vector form
%               or SPD matrices d x d x N in vector form
%   - S:        Base SPD matrix in vector form
%
% Returns:
%   - U:        Symmetric matrix Log_S(X) in vector form
%               or symmetric matrices d x d x N in vector form

X = vec2symmat(x);
S = vec2symmat(s);
U = logmap(X,S);
u = symmat2vec(U);
end