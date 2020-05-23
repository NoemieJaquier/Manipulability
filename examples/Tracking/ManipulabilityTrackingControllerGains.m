function ManipulabilityTrackingControllerGains
% This code shows how a robot matches a desired manipulability ellipsoid as 
% the main task (no desired position) using the formulation based the 
% manipulability Jacobien (Mandel notation).
% The matrix gain used for the manipulability tracking controller is now 
% defined as the inverse of a 2nd-order covariance matrix representing the 
% variability information obtained from a learning algorithm. 
% The user can:
%   1. Change the number of iterations
%   2. Choose different values for the covariance-based controller gain
%   3. Modify the robot kinematics by using the Robotics Toolbox 
%       functionalities
%   4. Change the initial conditions and desired manipulability ellipsoid

% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');


%% Auxiliar variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2;	% Time step
nbIter = 100; % Number of iterations
clrmap = lines(4);

% Initialisation of the covariance transformation
cov_2D(:,:,1) = [0.3 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3];  % Unitary gain (baseline)
cov_2D(:,:,2) = [0.03 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3]; % Priority on matching for 'x' axis
cov_2D(:,:,3) = [0.3 0.0 0.0; 0.0 0.03 0.0; 0.0 0.0 0.3]; % Priority on matching for 'y' axis
cov_2D(:,:,4) = [0.3 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.1]; % Correlation has highest priority for tracking

names = {'Baseline','Priority on x-axis','Priority on y-axis','Priority on xy-correltation'};

for n = 1:4
    
    %% Create robot
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Robot parameters
    nbDOFs = 4; %Nb of degrees of freedom
    armLength = 4; % Links length
    
    % Robot
    L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
    robot = SerialLink(repmat(L1,nbDOFs,1));
    q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
    J = robot.jacob0(q'); % Symbolic Jacobian
        
    % Define the desired manipulability
    q_Me_d = [0.0 ; pi/4 ; pi/2 ; pi/8];
    J_Me_d = robot.jacob0(q_Me_d); % Current Jacobian
    J_Me_d = J_Me_d(1:2,:);
    Me_d = (J_Me_d*J_Me_d');
    
    %% Testing Manipulability Transfer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initial conditions
    q0 =[pi/2 ; -pi/6; -pi/2 ; -pi/2]; % Initial robot configuration
    
    qt = q0;
    it = 1; % Iterations counter
    h1 = [];
    
    while( it < nbIter )
        delete(h1);
        
        Jt = robot.jacob0(qt); % Current Jacobian
        Jt_full = Jt;
        Jt = Jt(1:2,:);
        Htmp = robot.fkine(qt); % Forward Kinematics 
        Me_ct = (Jt*Jt'); % Current manipulability
        
        Me_track(:,:,it,n)=Me_ct;
        qt_track(:,it) = qt;
        
        % Compatibility with 9.X and 10.X versions of robotics toolbox
        if isobject(Htmp) % SE3 object verification
            xt = Htmp.t(1:2);
        else
            xt = Htmp(1:2,end);
        end
        xt_track(:,it,n) = xt;
        
        % Compute manipulability Jacobian
        Jm_t = compute_red_manipulability_Jacobian(Jt_full, 1:2);
        
        % Desired joint velocities
        M_diff = logmap(Me_d,Me_ct);
        dq_T1 = pinv(Jm_t)*(cov_2D(:,:,n)\symmat2vec(M_diff));
        
        % Updating joint position
        qt = qt + (dq_T1)*dt;
        it = it + 1; % Iterations
        
    end
    
    %% Final plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Robot and manipulability ellipsoids
    figure('position',[10 10 900 900],'color',[1 1 1]);
    hold on;
    p = [];
    for it = 1:2:nbIter-1
        colTmp = [.95,.95,.95] - [.8,.8,.8] * (it)/nbIter;
        p = [p; plotArm(qt_track(:,it), ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp)];
    end
    
    z = get(p,'ZData'); % to put arm plot down compared to GMM plot
    for i = 1:size(z,1)
        if isempty(z{i})
            set(p,'ZData',z{i}-10)
        end
    end
    
    plotGMM(xt(:,end), 1E-2*Me_d, [0.2 0.8 0.2], .4);
    plotGMM(xt_track(:,1,n), 1E-2*Me_track(:,:,1,n), [0.2 0.2 0.8], .4);
    plotGMM(xt(:,end), 1E-2*Me_ct, clrmap(n,:), .4);
    axis equal
    xlim([-3,9]); ylim([-4,8])
    set(gca,'fontsize',22,'xtick',[],'ytick',[]);
    xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
    title(names{n});
    drawnow;
end


%% Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manipulability ellipsoids

for n = 1:4
    figure('position',[10 10 1200 300],'color',[1 1 1]);
    hold on;
    for it = 1:10:nbIter-1
        plotGMM([it*dt;0], 2E-4*Me_d, [0.2 0.8 0.2], .1);
        plotGMM([it*dt;0], 2E-4*Me_track(:,:,it,n), clrmap(n,:), .3);
    end
    axis equal;
    set(gca,'fontsize',24,'ytick',[]);
    xlabel('$t$','fontsize',42,'Interpreter','latex');
    ylabel('$\mathbf{M}$','fontsize',42,'Interpreter','latex');
    title(names{n});
end

figure('position',[10 10 900 900],'color',[1 1 1]);
hold on;
for it = 1:2:nbIter-1
    for n = 1:4
        plotGMM(xt_track(:,it,n), 1E-2*Me_track(:,:,it,n), clrmap(n,:), .15);
    end
end
plotGMM(xt_track(:,1,1), 1E-2*Me_track(:,:,1,1), [0.1 0.1 0.4], .6);
axis equal;
set(gca,'fontsize',18);
xlabel('$x_1$','fontsize',30,'Interpreter','latex'); ylabel('$x_2$','fontsize',30,'Interpreter','latex');
title('End-effector and manipulability ellipsoids trajectories')
end
