function M = vec2symmat(v)
% Noémie Jaquier, 2018
%
% This function computes a SPD matrices based on a vector using Mandel
% notation.
%
% Parameters:
%   - v:        vectorized SPD matrix
%               or vectorized SPD matrices d' x N
% 
% Returns:
%   - M:        SPD matrix
%               or SPD matrices d x d x N
%

if ndims(v) == 1
    n = size(v,1);
    N = (-1 + sqrt(1 + 8*n))/2;
    M = diag(v(1:N));
    id = cumsum(fliplr(1:N));
    
    for i = 1:N-1
        M = M + diag(v(id(i)+1:id(i+1)),i)./sqrt(2) + diag(v(id(i)+1:id(i+1)),-i)./sqrt(2); % Mandel notation
        %   M = M + diag(v(id(i+1)+1:id(i+1)),i) + diag(v(id(i+1)+1:id(i+1)),-i); % Voigt notation
    end
else
    [d, N] = size(v);
    D = (-1 + sqrt(1 + 8*d))/2;
    for n = 1:N
        vn = v(:,n);
        Mn = diag(vn(1:D));
        id = cumsum(fliplr(1:D));
        
        for i = 1:D-1
            Mn = Mn + diag(vn(id(i)+1:id(i+1)),i)./sqrt(2) + diag(vn(id(i)+1:id(i+1)),-i)./sqrt(2); % Mandel notation
            %   M = M + diag(vn(id(i+1)+1:id(i+1)),i) + diag(vn(id(i+1)+1:id(i+1)),-i); % Voigt notation
        end
        M(:,:,n) = Mn;
    end
end
end
