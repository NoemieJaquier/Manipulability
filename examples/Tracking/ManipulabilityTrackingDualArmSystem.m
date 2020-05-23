function ManipulabilityTrackingDualArmSystem
% Noémie Jaquier and Leonel Rozo, 2018
%
% This code shows an example of manipulability tracking for a dual-arm
% system. The two robots work to match a desired dual-arm manipulability 
% ellipsoid in a master-slave principle. The main task of the left robot is
% to match the desired manipulability ellipsoid, while the main task of the
% right robot is to keep its end-effector at the same position as the
% first robot and its secondary task is to match the desired
% manipulability.
% The user can:
% 	1. Change the desired dual-arm manipulability ellipsoid

% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');

%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2;	% Time step
nbIter = 100; % Number of iterations
Kp = 30; % Gain for position control in task space
Km = 5; % Gain for manipulability control in task space

% Colors
clrmap = [  0.9970 0.6865 0.4692;
            0.1749 0.0670 0.3751;
            0.2 0.8 0.2];
        
%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robots parameters 
nbDOFs = 3; %Nb of degrees of freedom
armLength = 4; % Links length

% Robots
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot1 = SerialLink(repmat(L1,nbDOFs,1));

L2 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot2 = SerialLink(repmat(L2,nbDOFs,1));

% Initial robot configurations
q1_0 = [2*pi/3 ; -pi/3; -pi/2]; 
q2_0 = -q1_0 + [pi; 0; 0];
q0 = [q1_0; q2_0];

Htmp = robot1.fkine(q1_0); % Forward Kinematics
if isobject(Htmp) % SE3 object verification
	x1_0 = Htmp.t(1:2);
else
	x1_0 = Htmp(1:2,end);
end
Htmp = robot2.fkine(q2_0); % Forward Kinematics
if isobject(Htmp) % SE3 object verification
	x2_0 = Htmp.t(1:2);
else
	x2_0 = Htmp(1:2,end);
end

% Desired distance between the robots
dist_robots = x1_0(1)-x2_0(1);

% Grasp matrix for all robots
G = [eye(2) eye(2)];
pG = pinv(G);

% Desired Cartesian velocity
dxh = [0; 0]; 

% Desired manipulability ellipsoid
Me_d = [20 -20 ; -20 40];

%% Testing Manipulability Transfer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q1t = q1_0;
q2t = q2_0;
it = 1; % Iterations counter
h1 = [];
p1 = [];
p2 = [];
gmm_c = [];

figure('position',[10 10 1000 450],'color',[1 1 1]);

% Main control loop
while( it < nbIter )
	delete(h1);
    delete(p1);
    delete(p2);
	
    % First robot Jacobian, position and manipulability
	J1t = robot1.jacob0(q1t); % Current Jacobian
    J1t_full = J1t;
    J1t = J1t(1:2,:);
	Htmp = robot1.fkine(q1t); % Forward Kinematics (needed for plots)
	if isobject(Htmp) % SE3 object verification
		x1t = Htmp.t(1:2);
	else
		x1t = Htmp(1:2,end);
    end
    
    % Second robot Jacobian, position and manipulability
	J2t = robot2.jacob0(q2t); % Current Jacobian
    J2t_full = J2t;
    J2t = J2t(1:2,:);
	Htmp = robot2.fkine(q2t); % Forward Kinematics (needed for plots)
	if isobject(Htmp) % SE3 object verification
		x2t = Htmp.t(1:2);
	else
		x2t = Htmp(1:2,end);
    end
    
    % Desired robots positions
    x2d = x1t-[dist_robots;0];
    
    % Jacobian of the dual-arm system
    J = blkdiag(J1t,J2t);
    % Manipulability of the dual-arm system
    Me_ct = pG'*J*J'*pG;
    if it == 1
        % Save initial manipulability and position
        Me_ct0 = Me_ct;
        x1t0 = x1t;
    end

	% Compute manipulability Jacobian of the dual-arm system
	Jm_t = compute_red_manipulability_Jacobian_dual(J1t_full, J2t_full, 1:2, G);
	
    % Compute joint velocities for robot 1
    M_diff = logmap(Me_d,Me_ct);    
    dq_T = pinv(Jm_t)*Km*symmat2vec(M_diff);
    dq_T1 = dq_T(1:nbDOFs);
    
    % Compute joint velocities   for robot 2
    dxr2 = dxh + Kp*(x2d-x2t); % Reference task space velocity
    dq_T2 = pinv(J2t)*dxr2; % Main task joint velocities (given desired dx)
    % Redundancy resolution for robot 2
    M_command_ns = pinv(Jm_t)*symmat2vec(M_diff);
    dq_ns2 = (eye(nbDOFs) - pinv(J2t)*J2t) * Km * M_command_ns(nbDOFs+1:end); 
	
	% Plotting robot and manipulability ellipsoids
	subplot(1,2,1); 
	hold on;
	colTmp = [1,1,1] - [.8,.8,.8];
	p1 = plotArm(q1t, ones(nbDOFs,1)*armLength, [0; 0; 1], .2, colTmp);
    p2 = plotArm(q2t, ones(nbDOFs,1)*armLength, [dist_robots; 0; 1], .2, colTmp);
    if(it == 1)
		plotGMM(x1t, 1E-2*Me_d, clrmap(3,:), .3, '-.', 2, 1); % Scaled matrix!
        plotGMM(x1t, 1E-2*Me_ct0, clrmap(1,:), .4, '--', 2, 1); % Scaled matrix!
    end
    h1 = plotGMM(x1t, 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1); % Scaled matrix!
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
        plotGMM([0;0], 1E-2*Me_ct0, clrmap(1,:), .4, '--', 2, 1);
    end
    xlabel('$m_{11}$','fontsize',38,'Interpreter','latex');
    ylabel('$m_{22}$','fontsize',38,'Interpreter','latex');
	drawnow;
	
	% Updating joint position
	q1t = q1t + (dq_T1 )*dt;
    q2t = q2t + (dq_T2 + dq_ns2)*dt;
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
    p = [p; plotArm(q1t, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp)];
    p = [p; plotArm(q2t, ones(nbDOFs,1)*armLength, [dist_robots; 0; it*0.1], .2, colTmp)];
end

z = get(p,'ZData'); % to put arm plot down compared to GMM plot
for i = 1:size(z,1)
    if isempty(z{i})
        set(p,'ZData',z{i}-10)
    end
end

plotGMM(x1t0, 1E-2*Me_ct0, clrmap(1,:), .4, '--', 2, 1);
plotGMM(x1t, 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1);
plotGMM(x1t, 1E-2*Me_d, clrmap(3,:), .3, '-.', 2, 1);
axis equal
set(gca,'xtick',[],'ytick',[])
xlabel('$x_1$','fontsize',40,'Interpreter','latex'); ylabel('$x_2$','fontsize',40,'Interpreter','latex');

% Desired and initial manipulability ellipsoids
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .5, '-.', 3, 1);
plotGMM([0;0], 1E-2*Me_ct0, clrmap(1,:), .3, '--', 3, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1 1]);ylim([-1 1]);
set(gca,'xtick',[],'ytick',[]);
text(-.8,1,0,'Initial','FontSize',38,'Interpreter','latex')
axis equal;

% Desired and final manipulability ellipsoids
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .6, '-.', 3, 1);
plotGMM([0;0], 1E-2*Me_ct, clrmap(2,:), .3, '-', 3, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1 1]);ylim([-1 1]);
text(-.7,1,0,'Final','FontSize',38,'Interpreter','latex')
set(gca,'xtick',[],'ytick',[])
axis equal;

end




