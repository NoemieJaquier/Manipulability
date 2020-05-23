function Jm_red = compute_red_manipulability_Jacobian(J, taskVar, W)
% Noémie Jaquier, 2018
%
% This function computes the kinematic manipulability Jacobian (symbolic)
% in the form of a matrix using Mandel notation.
%
% Parameters:
%   - J:        Jacobian
%   - taskVar:  Task variables for which the manipulability Jacobian is
%               computed (optional, default: all)
%   - W:        Weight matrix whose elements correspond to the maximum 
%               joint velocities of the robot (optional, default: I)
% 
% Returns:
%   - Jm_red:   Kinematic manipulability Jacobian in matric form


if nargin < 2
    taskVar = 1:6;
end
if nargin < 3
    W = eye(size(J,2),size(J,2));
end

Jm = compute_manipulability_Jacobian(J,W);
Jm_red = [];
for i = 1:size(Jm,3)
	Jm_red = [Jm_red, symmat2vec(Jm(taskVar,taskVar,i))];
end

end