function ComDynamicManipulabilityTracking
% Leonel Rozo and Noémie Jaquier, 2020
%
% This code shows how a robot can match a desired center-of-mass dynamic
% manipulability ellipsoid as a main task (no desired position) using the 
% manipulability tracking formulation with the manipulability Jacobian 
% (Mandel notation). The manipulability definition uses the Jacobian 
% specified at the center of mass.
% The user can:
%   1. Use different controller gains for the manipulability tracking
% 	2. Change the initial conditions and desired manipulability ellipsoid
% 	3. Modify the robot kinematics by using the Robotics Toolbox
%      functionalities
%
% Reference for dynamic manipulability at the center of mass: 
%        Y. Gu et al, "Feasible Center of Mass Dynamic Manipulability of
%        humanoid robots" IEEE ICRA, 2015.

% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');

%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2;	% Time step
nbIter = 80; % Number of iterations
Km = 2; % Gain for manipulability control in task space
nbTaskVar = 2; % Planar task space

taskNumber = 1; % Options: 1 or 2

% Colors
clrmap = [  0.9970 0.6865 0.4692;
        0.1749 0.0670 0.3751;
        0.2 0.8 0.2];

%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot parameters
nbDOFs = 4; %Nb of degrees of freedom
armLength = 4;
armLengths = [armLength armLength armLength armLength]; % Links length
masses = [1 1 1 1]; % Links masses

% Robot
for i = 1:nbDOFs
    L1(i,1) = Link('d', 0, 'a', armLengths(i), 'alpha', 0, 'm', masses(i));
end
robot = SerialLink(L1);

% Robot CoM
totMass = sum(masses);
armLengthCoM = (masses.*armLengths./2 + (totMass-cumsum(masses)).*armLengths)./totMass;
for i = 1:nbDOFs
    L(i,1) = Link('d', 0, 'a', armLengthCoM(i), 'alpha', 0);
end
robotCoM = SerialLink(L);

q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
Jsym = jacob0(robotCoM, q', 'trans'); % Symbolic Jacobian for the CoM
Jsym = simplify(Jsym(1:nbTaskVar, :)); % 2D positional Jacobian
Isym = simplify(robot.inertia(q)); % Symbolic inertia

% Define the desired manipulability
if taskNumber == 1
    q_Me_d = [pi/16 ; pi/4 ; pi/8 ; -pi/8]; % task 1
else
    q_Me_d = [pi/2 ; -pi/6; -pi/2 ; -pi/2]; % task 2
end

JCoM_Me_d = jacob0(robotCoM, q_Me_d); % Current CoM Jacobian
JCoM_Me_d = JCoM_Me_d(1:2,:); % 2D positional Jacobian
I_Me_d = robot.inertia(q_Me_d'); % Current inertia matrix
LtJps = (JCoM_Me_d / I_Me_d); % Lambda(q) * pinv(J)
Me_d = LtJps*LtJps'; % Desired dynamic manipulability (T. Yoshikawa, 1985)

%% Compute the dynamic manipulability Jacobian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lsym = Jsym / Isym;

L_grad = [];
for i = 1 : length(q)
    L_grad = cat(3, L_grad, diff(Lsym, q(i)));
end

Jm = tmprod(permute(L_grad(1:nbTaskVar,:,:), [2,1,3]), Lsym(1:nbTaskVar,:), 1) + ...
    tmprod(L_grad(1:nbTaskVar,:,:), Lsym(1:nbTaskVar,:), 2);

Jm_red = [];
for i = 1:length(q)
    Jm_red = [Jm_red, symmat2vec(Jm(:,:,i))];
end

%Jmanip = matlabFunction(Jm_red);
Jmanip1 = matlabFunction(Jm_red(1,:));
Jmanip2 = matlabFunction(Jm_red(2,:));
Jmanip3 = matlabFunction(Jm_red(3,:));

%% Dynamic manipulability tracking at the CoM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
if taskNumber == 1
    q0 = [pi/2 ; -pi/6; -pi/2 ; -pi/2]; % Initial robot configuration task 1
else
    q0 = [0.0 ; pi/4 ; pi/2 ; pi/8]; % Initial robot configuration task 2
end

qt = q0;
it = 1; % Iterations counter
h1 = [];
gmm_c = [];

% Initial position x0
Htmp_CoM = robotCoM.fkine(q0); % Forward Kinematics (needed for plots)
% Current end-effector position
if isobject(Htmp_CoM) % SE3 object verification
    xCoM_0 = Htmp_CoM.t(1:2);
else
    xCoM_0 = Htmp_CoM(1:2,end);
end

figure('position',[10 10 1000 450],'color',[1 1 1]);
% Main control loop
while( it < nbIter )
    delete(h1);
    
    JCoM_t = robotCoM.jacob0(qt); % Current CoM Jacobian
    JCoM_t = JCoM_t(1:2,:);
    Htmp_CoM = robotCoM.fkine(qt); % Forward Kinematics for virtual CoM robot (needed for plots)
    
    It = robot.inertia(qt'); % Current inertia matrix
    IJ = (JCoM_t/It); %
    Me_ct = IJ*IJ'; % Current dynamic manipulability
    
    Me_track(:,:,it) = Me_ct;
    qt_track(:,it) = qt;
    
    % Current end-effector position
    if isobject(Htmp_CoM) % SE3 object verification
        xCoM_t = Htmp_CoM.t(1:2);
    else
        xCoM_t = Htmp_CoM(1:2,end);
    end
    
    % Compute manipulability Jacobian
    % 	Jm_t = Jmanip(qt(1),qt(2),qt(3));
    Jm_t = [Jmanip1(qt(1),qt(2),qt(3),qt(4)); ...
        Jmanip2(qt(1),qt(2),qt(3),qt(4));
        Jmanip3(qt(1),qt(2),qt(3),qt(4))];
    
    % Desired joint velocities
    M_diff = logmap(Me_d,Me_ct);
    dq_T1 = pinv(Jm_t)*Km*symmat2vec(M_diff);
    
    % Plotting robot and manipulability ellipsoids
    subplot(1,2,1);
    hold on;
    if(it == 1)
        plotGMM(xCoM_t, 100*Me_d, clrmap(3,:), .3, '-.', 2, 1); % Scaled matrix!
    end
    h1 = plotGMM(xCoM_t, 100*Me_ct, clrmap(2,:), .4, '-', 3, 1); % Scaled matrix!
    colTmp = [1,1,1] - [.8,.8,.8] * (it+10)/nbIter;
    plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp);
    axis square;
    axis equal;
    xlabel('$x_1$','fontsize',38,'Interpreter','latex');
    ylabel('$x_2$','fontsize',38,'Interpreter','latex');
    
    subplot(1,2,2);
    hold on; axis equal;
    delete(gmm_c);
    gmm_c = plotGMM([0;0], 100*Me_ct, clrmap(2,:), .4, '-', 3, 1); % Scaled matrix!
    if(it == 1)
        plotGMM([0;0], 100*Me_d, clrmap(3,:), .3, '-.', 2, 1); % Scaled matrix!
    end
    xlabel('$m_{11}$','fontsize',38,'Interpreter','latex');
    ylabel('$m_{22}$','fontsize',38,'Interpreter','latex');
    drawnow;
    
    % Updating joint position
    qt = qt + (dq_T1)*dt;
    it = it + 1; % Iterations++
    
end

%% Plots
figure('position',[10 10 900 900],'color',[1 1 1]);
hold on;
p = [];
for it = 1:2:nbIter-1
    colTmp = [1,1,1] - [.8,.8,.8] * (it)/nbIter;
    p = [p; plotArm(qt_track(:,it), armLengths, [0; 0; it*0.1], .2, colTmp)];
end
z = get(p,'ZData'); % to put arm plot down compared to GMM plot
for i = 1:size(z,1)
    if isempty(z{i})
        set(p,'ZData',z{i}-10)
    end
end
plotGMM(xCoM_0, 1E-2*Me_track(:,:,1), clrmap(1,:), .4, '--', 2, 1);
plotGMM(xCoM_t, 1E-2*Me_ct, clrmap(2,:), .4, '-', 3, 1);
plotGMM(xCoM_t, 1E-2*Me_d, clrmap(3,:), .3, '-.', 2, 1);
axis equal
set(gca,'xtick',[],'ytick',[])
xlabel('$x_1$','fontsize',40,'Interpreter','latex'); ylabel('$x_2$','fontsize',40,'Interpreter','latex');

figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 100*Me_d, clrmap(3,:), .3, '-.', 2, 1);
plotGMM([0;0], 100*Me_track(:,:,1), clrmap(1,:), .4, '--', 2, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); 
ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1.2 1.2]);ylim([-1.2 1.2]);
set(gca,'xtick',[],'ytick',[]);
text(-.8,1,0,'Initial','FontSize',38,'Interpreter','latex')
axis equal;

figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 100*Me_d, clrmap(3,:), .3, '-.', 2, 1);
plotGMM([0;0], 100*Me_ct, clrmap(2,:), .4, '-', 3, 1);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); 
ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1.2 1.2]);ylim([-1.2 1.2]);
text(-.7,1,0,'Final','FontSize',38,'Interpreter','latex')
set(gca,'xtick',[],'ytick',[])
axis equal;

figure()
hold on;
for it = 1:nbIter-1
    cost(it) = norm(logm(Me_d^-.5*Me_track(:,:,it)*Me_d^-.5),'fro');
end
plot([1:nbIter-1].*dt, cost, '-','color',[0 0 .7],'Linewidth',3);
set(gca,'fontsize',14);
xlim([0 nbIter*dt])
xlabel('$t$','fontsize',22,'Interpreter','latex'); 
ylabel('$d(\hat{\mathbf{M}}_t,\mathbf{M}_t)$','fontsize',22,'Interpreter','latex');

end

