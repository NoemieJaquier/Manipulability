function Cost = ManipulabilitySteinDivergenceCost(Me_c, Me_d)
% Leonel Rozo, 2017
%
% This function computes a symbolic equation representing cost function 
% that evaluates how similar two manipulability ellipsoids are based on the
% Stein divergence.
%
% Parameters:
%   - Me_c:     Manipulability ellipsoid at current time step (symbolic)
%   - Me_d:     Desired manipulability ellipsoid
% 
% Returns:
%   - Cost:    Symbolic equation of the cost function


%% Computation of cost function and its gradient
% Computation of the cost function
[eVc, ~] = eig(Me_d);
des_vec = eVc(:,1)/norm(eVc(:,1)); % Desired VME major axis

C = log(det(0.5*(Me_d+Me_c))) - 0.5*log(det(Me_d*Me_c));

% Creating a MATLAB function from symbolic variable
Cost =  matlabFunction(C);
end