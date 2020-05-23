function Jm = compute_manipulability_Jacobian(J,W)
% Noémie Jaquier, 2018
%
% This function computes the kinematic manipulability Jacobian (symbolic).
%
% Parameters:
%   - J:        Jacobian
%   - W:        Weight matrix whose elements correspond to the maximum 
%               joint velocities of the robot (optional, default: I)
% 
% Returns:
%   - Jm:       Kinematic manipulability Jacobian

if nargin < 2
    W = eye(size(J,2),size(J,2));
end

J_grad = compute_joint_derivative_Jacobian(J);

Jm = tmprod(J_grad,J*(W*W'),2) + tmprod(permute(J_grad,[2,1,3]),J*(W*W'),1);

end