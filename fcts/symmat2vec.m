function v = symmat2vec(M)
% Noémie Jaquier, 2018
%
% This function computes a vectorization of SPD matrices using Mandel
% notation.
%
% Parameters:
%   - M:        SPD matrix
%               or SPD matrices d x d x N
% 
% Returns:
%   - v:        vectorized SPD matrix
%               or vectorized SPD matrices d' x N
%

if ndims(M) == 2
    N = size(M,1);
    v = [];
    
    v = diag(M);
    for n = 1:N-1
        v = [v; sqrt(2).*diag(M,n)]; % Mandel notation
        %   v = [v; diag(M,n)]; % Voigt notation
    end
else
    [D, ~, N] = size(M);
    
    v = [];
    for n = 1:N
        vn = [];
        vn = diag(M(:,:,n));
        for d = 1:D-1
            vn = [vn; sqrt(2).*diag(M(:,:,n),d)]; % Mandel notation
            %   v = [v; diag(M,n)]; % Voigt notation
        end
        v = [v vn];
    end
    
end
end
