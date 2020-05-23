function ManipulabilityTrackingWithNullspace
% Noémie Jaquier and Leonel Rozo, 2018
%
% This code shows how a robot can match a desired manipulability ellipsoid
% as the main task, while keeping a desired joint configuration as 
% secondary task (by using the nullspace of the manipulability Jacobian).
% The user can:
%   1. Use different controller gains for the manipulability tracking
%   2. Use different controller gains for the secondary task
%   3. Change which joint angle should remain fixed as a secondary task 
%       (by modifying Wq)
%   4. Activate or deactivate the nullspace controller for the secondary
%       task.
% 	5. Change the initial conditions and desired manipulability ellipsoid
% 	6. Modify the robot kinematics by using the Robotics Toolbox
%      functionalities
%
% First run 'startup_rvc' from the robotics toolbox

% Holding a manipulability ellipsoid (position) as main task, while keeping
% a desired joint configuration as secondary task
%

% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');

%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2;	% Time step
nbIter = 70; % Number of iterations
Km = 3; % Gain for manipulability control in task space
KnP = 20; % P gain for joint position control in null space
KnD = 2; % D gain for joint position control in null space
taskVar = 1:2;

% Keep or not a desired joint angles as secondary task
nullspaceCommandActivated = 1; 

%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot parameters 
nbDOFs = 6; %Nb of degrees of freedom
armLength = 3; % Links length

% Robot
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
J = robot.jacob0(q'); % Symbolic Jacobian

% Define the desired manipulability
q_Me_d = [pi/2 ; pi/4 ; pi/3 ; -pi/8; pi/3; pi/5]; 

J_Me_d = robot.jacob0(q_Me_d); % Current Jacobian
J_Me_d = J_Me_d(1:2,:);
MeP_d = (J_Me_d*J_Me_d');

% Define which joint angles should remain fixed (here: first joint)
Wq = 1E0*diag([1.0 ; zeros(nbDOFs-1,1)]);

%% Testing Manipulability Tracking with joint position control in nullspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
q0 = [pi/3 ; -pi/6; 0 ; pi/2; pi/3; pi/3]; % Initial robot configuration 
qt = q0;
qh = q0;
dqt = zeros(nbDOFs,1); % Desired joint velocity
it = 1; % Iterations counter
dMeP_d = zeros(length(taskVar)); % Desired manipulability velocity (position)
h1 = []; % for plots
gmm_c = []; % for plots

% Initial end-effector position (compatibility with 9.X and 10.X versions
% of robotics toolbox)
Htmp = robot.fkine(q0); % Forward Kinematics
if isobject(Htmp) % SE3 object verification
	x0 = Htmp.t(1:2);
else
	x0 = Htmp(1:2,end);
end

figure('position',[10 10 1000 450],'color',[1 1 1]);

while( it < nbIter )
    if ~mod(it,10) || it == 1
        delete(h1);
    end
    
    Jt_full = robot.jacob0(qt); % Current Jacobian
    JtP = Jt_full(taskVar,:);
    Htmp = robot.fkine(qt); % Forward Kinematics (needed for plots)
    MeP_ct = (JtP*JtP'); % Current manipulability
        
    % Current end-effector position
    if isobject(Htmp) % SE3 object verification
      xt = Htmp.t(1:2);
    else
      xt = Htmp(1:2,end);
    end
    % Manipulability Jacobian
    JmP_t = compute_red_manipulability_Jacobian(Jt_full, taskVar);
        
    % Desired joint velocities
    MeP_diff = logmap(MeP_d,MeP_ct);
        
    % Singularity avoidance
    minEigJmP = min(svd(JmP_t));
    threshold = 1e-2;
    if minEigJmP < threshold
        pinvJmP_t = (JmP_t'*JmP_t + threshold.*eye(nbDOFs)) \ JmP_t';
    else
        pinvJmP_t = pinv(JmP_t);%(Jm_t'*Jm_t) \ Jm_t';
    end
    
    % Reference task space velocity
    dMeP = symmat2vec(dMeP_d) + Km*symmat2vec(MeP_diff);
    % Main task joint velocities
    dq_T1 = pinvJmP_t*dMeP; 
    % Redundancy resolution
    if nullspaceCommandActivated
        M_command_ns = Wq*(KnP*(qh - qt) - KnD*dqt);
        dq_ns = (eye(robot.n) - pinvJmP_t*JmP_t) * M_command_ns; 
    end
    
    % Plotting robot and manipulability ellipsoids
    subplot(1,2,1); hold on;
    delete(h1)
    if(it == 1)
      plotGMM(xt, 1E-2*MeP_d, [0.2 0.8 0.2], .4, '-.', 3, 1); % Scaled matrix!
    end
    h1 = plotGMM(xt, 1E-2*MeP_ct, [0.1749 0.0670 0.3751], .4, '-', 3, 1); % Scaled matrix!
    colTmp = [1,1,1] - [.8,.8,.8] * (it+10)/nbIter;
    plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp);
    axis square; axis equal;
    xlabel('$x_1$','fontsize',38,'Interpreter','latex');
    ylabel('$x_2$','fontsize',38,'Interpreter','latex');
	
    subplot(1,2,2); hold on; axis equal;
    delete(gmm_c);
    gmm_c = plotGMM([0;0], 1E-2*MeP_ct, [0.1749 0.0670 0.3751], .4, '-', 3, 1); % Scaled matrix!
    if(it == 1)
      plotGMM([0;0], 1E-2*MeP_d, [0.2 0.8 0.2], .4, '-.', 3, 1); % Scaled matrix!
    end
    xlabel('$m_{11}$','fontsize',38,'Interpreter','latex');
    ylabel('$m_{22}$','fontsize',38,'Interpreter','latex');
    drawnow;
    
    % Updating joint position
    if nullspaceCommandActivated
        dqt = (dq_T1 + dq_ns);
    else
        dqt = (dq_T1);
    end
    qt = qt + dqt*dt;
    it = it + 1; % Iterations++
    
end

%% Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot and manipulability ellipsoids
figure('position',[10 10 1200 600],'color',[1 1 1]);
hold on;
% Desired first join configuration in green
p = plotArm(qh(1), ones(1,1)*armLength, [0; 0; 0], .2, [0.26 0.70 0.5]);
% Final robot posture
plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; nbIter*0.1], .2, colTmp);
% Desired and final manipulability ellipsoids
plotGMM(xt, 1E-2*MeP_d, [0.2 0.8 0.2], .4, '-.', 3, 1);
plotGMM(xt, 1E-2*MeP_ct, [0.1749 0.0670 0.3751], .4, '-', 3, 1);

z = get(p,'ZData'); % to put green arm plot up compared to black one
for i = 1:size(z,1)
	if ~isempty(z{i})
        set(p,'ZData',z{i}+40) 
	end
end
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); 
ylabel('$x_2$','fontsize',38,'Interpreter','latex');
axis equal
set(gca,'xtick',[],'ytick',[])
end