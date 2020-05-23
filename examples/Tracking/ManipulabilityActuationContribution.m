function ManipulabilityActuationContribution
% Noémie Jaquier and Leonel Rozo, 2020
%
% This code illustrates the effect of actuation contribution on the shape 
% of the manipulability ellipsoid. A weight matrix representing the maximum
% joint velocity is added to the definition of the velocity manipulability.
% The user can:
%   1. Use different weight matrices corresponding to different maximum
%   joint velocities for the robot
%   2. Display the kinematic or force manipulability ellipsoids
% 	3. Change the initial conditions
% 	4. Modify the robot kinematics by using the Robotics Toolbox
%      functionalities
%
% Note 1: The kinematic manipulability with actuation contribution can be
% computed as J*(W*W')*J', with J the Jacobian and W the weight matrix 
% representing the maximum joint velocity (as shown in the code below).
% Note 2: In order to track a manipulability ellipsoid including the
% actuation contribution, the kinematic manipulability Jacobian is computed
% as Jm_t = compute_red_manipulability_Jacobian(J, taskVar, W), with J the
% Jacobian, taskVar the indices of the task (1:2 for planar robots) and W
% the weight matrix representing the maximum joint velocity. 
%
% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');

%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
manipulabilityType = 'kinematic'; % Options: 'kinematic' or 'force'

%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Robot parameters 
nbDOFs = 4; %Nb of degrees of freedom
armLength = 4; % Links length
% Robot
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));

%% Weight matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W(:,:,1) = diag([5.0, 1.0, 1.0, 1.0]);
W(:,:,2) = diag([1.0, 5.0, 1.0, 1.0]);
W(:,:,3) = diag([1.0, 1.0, 5.0, 1.0]);
W(:,:,4) = diag([1.0, 1.0, 1.0, 5.0]);
Wb = diag([1.0, 1.0, 1.0, 1.0]); % baseline

%% Computing manipulability ellipsoid for different weight matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
q0 = [0.0 ; pi/4 ; pi/2 ; pi/8];

Jt = robot.jacob0(q0); % Current Jacobian
Jt = Jt(1:2,:);
Htmp = robot.fkine(q0); % Forward Kinematics (needed for plots)

% Compatibility with 9.X and 10.X versions of robotics toolbox
if isobject(Htmp) % SE3 object verification
    xt = Htmp.t(1:2);
else
    xt = Htmp(1:2,end);
end

% Manipulability ellipsoids
for i = 1:size(W,3)
    if strcmp(manipulabilityType, 'kinematic')
        Me(:,:,i) = Jt*(W(:,:,i)*W(:,:,i)')*Jt'; % Kinematic manipulability
    else
        Me(:,:,i) = inv(Jt/(W(:,:,i)*W(:,:,i)')*Jt'); % Force manipulability
    end
end

if strcmp(manipulabilityType, 'kinematic')
    Meb = Jt*(Wb*Wb')*Jt'; % Kinematic manipulability
else
    Meb = inv(Jt/(Wb*Wb')*Jt'); % Force manipulability
end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(size(Me,3));

figure('position',[10 10 900 900],'color',[1 1 1]);
hold on;
plotArm(q0, ones(nbDOFs,1)*armLength, [0; 0; 0.1], .2, [.2 .2 .2]);

for i = 1:size(Me,3)
    if strcmp(manipulabilityType, 'kinematic')
        % plot for kinematic manipulability
        plotGMM(xt, 0.01*Me(:,:,i), clrmap(i,:), .4, '-', 3, 1); 
    else
        % plot for force manipulability
        plotGMM(xt, 100*Me(:,:,i), clrmap(i,:), .4, '-', 3, 1); 

    end
end
if strcmp(manipulabilityType, 'kinematic')
    % plot for kinematic manipulability
    plotGMM(xt, 0.01*Meb, [.7, .7, .7], .4, '-', 3, 1); 
else
    % plot for force manipulability
    plotGMM(xt, 100*Meb, [.5, .5, .5], .4, '-', 3, 1); 
end
axis equal
set(gca,'xtick',[],'ytick',[])
xlabel('$x_1$','fontsize',40,'Interpreter','latex'); 
ylabel('$x_2$','fontsize',40,'Interpreter','latex');

end
