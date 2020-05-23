function ManipulabilityTrackingSecondaryTask
% Noémie Jaquier and Leonel Rozo, 2018
%
% This code implements a manipulability tracking task as a secondary 
% objective. Here, the robot is required to hold a desired Cartesian 
% position as main task, while matching a desired manipulability
% ellipsoid as secondary task using the manipulability Jacobian 
% formulation (Mandel notation). 
% The user can:
%     1. Use different controller gains for both the position controller 
%         and the manipulability tracking
%     2. Change the initial conditions, desired Cartesian position 
%         and desired manipulability ellipsoid
%     3. Modify the robot kinematics by using the Robotics Toolbox 
%         functionalities
%
% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');

%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2;	% Time step
nbIter = 50; % Number of iterations
Kp = 8; % Gain for position control in task space
Km = 5; % Gain for manipulability control in nullspace
taskNumber = 1; % Options: 1 or 2

% Colors
clrmap = [  0.9970 0.6865 0.4692; 
            0.1749 0.0670 0.3751;
            0.2 0.8 0.2];
        
%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot parameters
nbDOFs = 4; % Nb of degrees of freedom
armLength = 4; % Links length

% Robot
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
J = robot.jacob0(q'); % Symbolic Jacobian

% Define the desired manipulability
if taskNumber == 1
    Me_d = [20 -40 ; -40 150]; % task1
else
    q_Me_d = [pi/2 ; -pi/6; -pi/2 ; -pi/2]; % task 2 
    J_Me_d = robot.jacob0(q_Me_d); % Current Jacobian
    J_Me_d = J_Me_d(1:2,:);
    Me_d = (J_Me_d*J_Me_d');
end

%% Testing Manipulability Transfer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
dxh = [0; 0]; % Desired Cartesian velocity

% Initial robot configuration
if taskNumber == 1
    q0 = [-pi/2 ; pi/6; pi/2 ; 1.4]; % task 1
else
    q0 = [pi/16 ; -pi/4; -pi/2 ; -pi/2]; % task 2
end

qt = q0;
xd = zeros(2,1);
it = 1; % Iterations counter
h1 = [];
gmm_c = [];

% Main control loop
figure('position',[10 10 1000 450],'color',[1 1 1]);
while( it < nbIter )
    delete(h1);
    
    Jt = robot.jacob0(qt); % Current Jacobian
    Jt_full = Jt;
    Jt = Jt(1:2,:);
    Htmp = robot.fkine(qt); % Forward Kinematics (needed for plots)
    Me_ct = (Jt*Jt'); % Current manipulability
    
    Me_track(:,:,it) = Me_ct;
    qt_track(:,it) = qt;
    
    % Current end-effector position (compatibility with 9.X and 10.X versions
    % of robotics toolbox)
    if isobject(Htmp) % SE3 object verification
        xt = Htmp.t(1:2);
    else
        xt = Htmp(1:2,end);
    end
    
    % Setting desired task position as the first end-effector position
    if(it == 1)
        xd = xt;
    end
    
    % Compute manipulability Jacobian
    Jm_t_red = compute_red_manipulability_Jacobian(Jt_full, 1:2);
    
    % Compute joint velocities  
    dxr = dxh + Kp*(xd-xt); % Reference task space velocity
    dq_T1 = pinv(Jt)*dxr; % Main task joint velocities (given desired dx)
    
    % Compute nullspace joint velocities  
    M_diff = logmap(Me_d,Me_ct);    
    M_command_ns = pinv(Jm_t_red)*symmat2vec(M_diff);
    dq_ns = (eye(nbDOFs) - pinv(Jt)*Jt) * Km * M_command_ns; % Redundancy resolution
    
    % Plotting robot and manipulability ellipsoids
    subplot(1,2,1); hold on;
    if(it == 1)
        plotGMM(xt, 1E-2*Me_d, clrmap(3,:), .5, '-.', 3, 1); % Scaled matrix!
    end
    h1 = plotGMM(xt, 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1); % Scaled matrix!
    colTmp = [1,1,1] - [.7,.7,.7] * it/nbIter;
    plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp);
    axis square;
    axis equal;
    xlabel('$x_1$','fontsize',38,'Interpreter','latex'); 
    ylabel('$x_2$','fontsize',38,'Interpreter','latex');
    
    subplot(1,2,2);
    hold on; axis equal;
    delete(gmm_c);
    gmm_c = plotGMM([0;0], 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1); % Scaled matrix!
    if(it == 1)
        plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .3, '-.', 2, 1); % Scaled matrix!
    end
    xlabel('$m_{11}$','fontsize',38,'Interpreter','latex');
    ylabel('$m_{22}$','fontsize',38,'Interpreter','latex');
    drawnow;
    
    % Updating joint position
    qt = qt + (dq_T1 + dq_ns)*dt;
    it = it + 1; % Iterations++
    
end

%% Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot and manipulability ellipsoids
figure('position',[10 10 900 900],'color',[1 1 1]);
hold on;
p = [];
for it = 1:2:nbIter-1
	colTmp = [1,1,1] - [.8,.8,.8] * (it)/nbIter;
	p = [p; plotArm(qt_track(:,it), ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp)];
end

z = get(p,'ZData'); % to put arm plot down compared to GMM plot
for i = 1:size(z,1)
	if isempty(z{i})
	set(p,'ZData',z{i}-10) 
	end
end

plotGMM(xt, 1E-2*Me_d, clrmap(3,:), .4, '-.', 3, 1);
plotGMM(xt, 1E-2*Me_track(:,:,1), clrmap(1,:), .4, '--', 3, .8);
plotGMM(xt, 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1);
axis equal
set(gca,'xtick',[],'ytick',[]);
xlabel('$x_1$','fontsize',40,'Interpreter','latex'); ylabel('$x_2$','fontsize',40,'Interpreter','latex');

% Desired and initial manipulability ellipsoids
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .5, '-.', 3, 1);
plotGMM([0;0], 1E-2*Me_track(:,:,1), clrmap(1,:), .4, '--', 3, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
text(0,1,0,'Initial','FontSize',38,'Interpreter','latex')
set(gca,'xtick',[],'ytick',[]);
axis equal;

% Desired and final manipulability ellipsoids
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .5, '-.', 3, 1);
plotGMM([0;0], 1E-2*Me_ct, clrmap(2,:), .5, '-', 3, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
text(0.2,1,0,'Final','FontSize',38,'Interpreter','latex')
set(gca,'xtick',[],'ytick',[])
axis equal;

end
