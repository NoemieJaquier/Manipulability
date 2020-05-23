function [S,iperm] = tmprod(T,U,mode)
% Noémie Jaquier, 2018
%
% This function computes the mode-n tensor-matrix product of a tensor T and
% matrix U.
%
% Parameters:
%   - T:        Tensor
%   - U:        Matrix
%   - mode:     Mode of the product
% 
% Returns:
%   - S:        Mode-n tensor matrix product
%   - iperm:    Permutation operation
%
% This function has been adapted from the implementation in Tensorlab by:
%            Laurent Sorber (Laurent.Sorber@cs.kuleuven.be),
%            Nick Vannieuwenhoven (Nick.Vannieuwenhoven@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

% Tensor size
size_tens = ones(1,mode);
size_tens(1:ndims(T)) = size(T);
N = length(size_tens);

% Compute the complement of the set of modes.
bits = ones(1,N);
bits(mode) = 0;
modec = 1:N;
modec = modec(logical(bits(modec)));

% Permutation of the tensor
perm = [mode modec];
size_tens = size_tens(perm);
S = T; 
if mode ~= 1
    S = permute(S,perm); 
end

% n-mode product
size_tens(1) = size(U,1);
S = reshape(U*reshape(S,size(S,1),[]),size_tens);

% Inverse permutation
iperm(perm(1:N)) = 1:N;
S = permute(S,iperm); 

end
