function Jm_red = compute_red_manipulability_Jacobian_dual(J1,J2,taskVar,G)
% Noémie Jaquier, 2020
%
% This function computes the kinematic manipulability Jacobian (symbolic)
% of a dual-arm system in the form of a matrix using Mandel notation.
% The arms are considered to have independent kinematics.
%
% Parameters:
%   - J1:       Jacobian of arm 1
%   - J2:       Jacobian of arm 2
%   - taskVar:  Task variables for which the manipulability Jacobian is
%               computed (optional, default: all)
%   - G:        Grasp matrix
% 
% Returns:
%   - Jm_red:   Kinematic manipulability Jacobian in matric form

nbVar = length(taskVar);
nbDof1 = size(J1,2);
nbDof2 = size(J2,2);

% Jacobian of dual-arm system
J = blkdiag(J1(taskVar,:),J2(taskVar,:));

% Gradient of dual-arm system
J1_grad = compute_joint_derivative_Jacobian(J1);
J2_grad = compute_joint_derivative_Jacobian(J2);
J_grad = zeros(2*nbVar, nbDof1+nbDof2, nbDof1+nbDof2);
J_grad(1:nbVar, 1:nbDof1, 1:nbDof1) = J1_grad(taskVar,:,:);
J_grad(nbVar+1:end, nbDof1+1:end, nbDof1+1:end) = J2_grad(taskVar,:,:);

% Inverse of grasp matrix
pG = pinv(G);

% Manipulability Jacobian of the dual-arm system
Jm = tmprod(tmprod(J_grad,pG',1),pG'*J,2) + ...
    tmprod(tmprod(permute(J_grad,[2,1,3]),pG'*J,1),pG',2);

Jm_red = [];
for i = 1:size(Jm,3)
	Jm_red = [Jm_red, symmat2vec(Jm(:,:,i))];
end

end