function ManipulabilityLearning
% Noémie Jaquier and Leonel Rozo, 2020
%
% This code shows how a robot builds models to follow a desired Cartesian
% trajectory while matching a desired profile of manipulability ellipsoids 
% over time. 
% The learning framework is built on two GMMs, one for encoding the
% demonstrated Cartesian trajectories, and the other one for encoding 
% the profiles of manipulability ellipsoids observed during the 
% demonstrations. The former is a classic GMM, while the latter is a 
% GMM that relies on an SPD-matrices manifold formulation.
% The desired trajectoried for reproducing the task are constructed with
% GMR based on the two GMM models.
% 
% The demonstrations are generated with a 3-DoFs planar robot that follows
% a set of Cartesian trajectories. 
% To see how a robot can then reproduce the obtained desired trajectories,
% see the examples in the Transfer folder.
% 
% The user can:
%   1. Define the number of states of the models
%   2. Modify the robot kinematics by using the Robotics Toolbox 
%       functionalities

% First run 'startup_rvc' from the robotics toolbox

addpath('../../fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 100; % Number of datapoints in a trajectory
nbSamples = 4; % Number of demonstrations
nbIter = 10; % Number of iteration for the Gauss Newton algorithm (Riemannian manifold)
nbIterEM = 10; % Number of iteration for the EM algorithm
letter = 'C'; % Letter to use as dataset for demonstration data

modelPD.nbStates = 5; %Number of Gaussians in the GMM over man. ellipsoids
modelPD.nbVar = 3; % Dimension of the manifold and tangent space (1D input + 2^2 output)
modelPD.nbVarOut = 2; % Dimension of the output
modelPD.nbVarOutVec = modelPD.nbVarOut + modelPD.nbVarOut*(modelPD.nbVarOut-1)/2; % Dimension of the output in vector form
modelPD.nbVarVec = modelPD.nbVar - modelPD.nbVarOut + modelPD.nbVarOutVec; % Dimension of the manifold and tangent space in vector form
modelPD.nbVarCovOut = modelPD.nbVar + modelPD.nbVar*(modelPD.nbVar-1)/2; %Dimension of the output covariance
modelPD.dt = 1E-2; % Time step duration
modelPD.params_diagRegFact = 1E-4; % Regularization of covariance
modelPD.Kp = 100; % Gain for position control in task space

modelKin.nbStates = 5; % Number of states in the GMM over 2D Cartesian trajectories
modelKin.nbVar = 3; % Number of variables [t,x1,x2]
modelKin.dt = modelPD.dt; % Time step duration

%% Create robots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robots parameters
nbDOFs = 3; % Nb of degrees of freedom for teacher robot
% armLength = 4; % For I and L
armLength = 5; % For C
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robotT = SerialLink(repmat(L1,nbDOFs,1)); % Robot teacher
q0T = [pi/4 0.0 -pi/9]; % Initial robot configuration

%% Load handwriting data and generating manipulability ellipsoids
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Loading demonstration data...');
load(['../../data/2Dletters/' letter '.mat'])

xIn(1,:) = (1:nbData) * modelPD.dt; % Time as input variable
X = zeros(3,3,nbData*nbSamples); % Matrix storing t,x1,x2 for all the demos
X(1,1,:) = reshape(repmat(xIn,1,nbSamples),1,1,nbData*nbSamples); % Stores input
Data=[];

for n=1:nbSamples
    s(n).Data=[];
    dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
    s(n).Data = [s(n).Data; dTmp];
    
    % Obtain robot configurations for the current demo given initial robot pose q0
    T = transl([s(n).Data(1:2,:) ; zeros(1,nbData)]');
    
    % One way to check robotics toolbox version
    if isobject(robotT.fkine(q0T))  % 10.X
        maskPlanarRbt = [ 1 1 0 0 0 0 ];  % Mask matrix for a 3-DoFs robots for position (x,y)
        q = robotT.ikine(T, q0T', 'mask', maskPlanarRbt)';  % Based on an initial pose
    else  % 9.X
        maskPlanarRbt = [ 1 1 1 0 0 0 ];
        q = robotT.ikine(T, q0T', maskPlanarRbt)'; % Based on an initial pose
    end
    s(n).q = q; % Storing joint values
    
    % Computing force/velocity manipulability ellipsoids, that will be later
    % used for encoding a GMM in the force/velocity manip. ellip. manifold
    for t = 1 : nbData
        auxJ = robotT.jacob0(q(:,t),'trans');
        J = auxJ(1:2,:);
        X(2:3,2:3,t+(n-1)*nbData) = J*J'; % Saving ME
    end
    Data = [Data [xIn ; s(n).Data]]; % Storing time and Cartesian positions
end
% SPD data in vector shape
x = [reshape(X(1,1,:),1,nbData*nbSamples); symmat2vec(X(2:end,2:end,:))];

%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning GMM1 (2D Cartesian position)...');
modelKin = init_GMM_timeBased(Data, modelKin); % Model for position
modelKin = EM_GMM(Data, modelKin);

disp('Learning GMM2 (Manipulability ellipsoids)...');
% Initialisation on the manifold
in=1; outMat=2:modelPD.nbVar; out = 2:modelPD.nbVarVec;
modelPD = spd_init_GMM_kbins(x, modelPD, nbSamples,out);
modelPD.Mu = zeros(size(modelPD.MuMan));
L = zeros(modelPD.nbStates, nbData*nbSamples);
xts = zeros(modelPD.nbVarVec, nbData*nbSamples, modelPD.nbStates);

% EM for SPD matrices manifold
for nb=1:nbIterEM
    fprintf('.');
    % E-step
    for i=1:modelPD.nbStates
        xts(in,:,i) = x(in,:)-repmat(modelPD.MuMan(in,i),1,nbData*nbSamples);
        xts(out,:,i) = logmap_vec(x(out,:), modelPD.MuMan(out,i));
        L(i,:) = modelPD.Priors(i) * gaussPDF(xts(:,:,i), modelPD.Mu(:,i), modelPD.Sigma(:,:,i));
    end
    % Responsibilities
    GAMMA = L ./ repmat(sum(L,1)+realmin, modelPD.nbStates, 1);
    H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
    
    % M-step
    for i=1:modelPD.nbStates
        % Update Priors
        modelPD.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
        % Update MuMan
        for n=1:nbIter
            % Update on the tangent space
            uTmp = zeros(modelPD.nbVarVec,nbData*nbSamples);
            uTmp(in,:) = x(in,:) - repmat(modelPD.MuMan(in,i),1,nbData*nbSamples);
            uTmp(out,:) = logmap_vec(x(out,:), modelPD.MuMan(out,i));
            uTmpTot = sum(uTmp.*repmat(H(i,:),modelPD.nbVarVec,1),2);
            % Update on the manifold
            modelPD.MuMan(in,i) = uTmpTot(in,:) + modelPD.MuMan(in,i);
            modelPD.MuMan(out,i) = expmap_vec(uTmpTot(out,:), modelPD.MuMan(out,i));
        end
        % Update Sigma
        modelPD.Sigma(:,:,i) = uTmp * diag(H(i,:)) * uTmp' + eye(modelPD.nbVarVec) .* modelPD.params_diagRegFact;
    end
end
% Eigendecomposition of Sigma
for i=1:modelPD.nbStates
    [V(:,:,i), D(:,:,i)] = eig(modelPD.Sigma(:,:,i));
end

%% GMR (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Regression...');
% Inputs
xIn = zeros(1,nbData);
xIn(1,:) = (1:nbData) * modelPD.dt;

in=1; % Input dimension
out=2:modelPD.nbVarVec; % Output dimensions for GMM over manipulabilities
nbVarOut = length(out);
outMan = 2:modelPD.nbVar;

% Initialisations for GMR for manipulability ellipsoids
uhat = zeros(nbVarOut,nbData);
xhat = zeros(nbVarOut,nbData);
uOut = zeros(nbVarOut,modelPD.nbStates,nbData);
expSigma = zeros(nbVarOut,nbVarOut,nbData);
H = [];

for t=1:nbData
    % GMR for 2D Cartesian trajectory
    [xd(:,t), sigma_xd(:,:,t)] = GMR(modelKin, t*modelKin.dt, in, 2:modelKin.nbVar);
    
    % GMR for manipulability ellipsoids
    % Compute activation weight
    for i=1:modelPD.nbStates
        H(i,t) = modelPD.Priors(i) * gaussPDF(xIn(:,t)-modelPD.MuMan(in,i),...
            modelPD.Mu(in,i), modelPD.Sigma(in,in,i));
    end
    H(:,t) = H(:,t) / sum(H(:,t)+realmin);
    % Compute conditional mean (with covariance transportation)
    if t==1
        [~,id] = max(H(:,t));
        xhat(:,t) = modelPD.MuMan(out,id); % Initial point
    else
        xhat(:,t) = xhat(:,t-1);
    end
    % Iterative computation
    for n=1:nbIter
        uhat(:,t) = zeros(nbVarOut,1);
        for i=1:modelPD.nbStates
            % Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
            S1 = vec2symmat(modelPD.MuMan(out,i));
            S2 = vec2symmat(xhat(:,t));
            Ac = blkdiag(1,transp_operator(S1,S2));
            % Parallel transport of eigenvectors
            for j = 1:size(V,2)
                vMat(:,:,j,i) = blkdiag(blkdiag(V(in,j,i)),vec2symmat(V(out,j,i)));
                pvMat(:,:,j,i) = Ac * D(j,j,i)^.5 * vMat(:,:,j,i) * Ac';
                pV(:,j,i) = [diag(pvMat(in,in,j,i)); symmat2vec(pvMat(outMat,outMat,j,i))];
            end
            % Parallel transported sigma (reconstruction from eigenvectors)
            pSigma(:,:,i) = pV(:,:,i)*pV(:,:,i)';
            % Gaussian conditioning on the tangent space
            uOut(:,i,t) = logmap_vec(modelPD.MuMan(out,i), xhat(:,t)) + ...
                pSigma(out,in,i)/pSigma(in,in,i)*(xIn(:,t)-modelPD.MuMan(in,i));
            
            uhat(:,t) = uhat(:,t) + uOut(:,i,t) * H(i,t);
        end
        % Projection back onto the manifold
        xhat(:,t) = expmap_vec(uhat(:,t), xhat(:,t));
    end 
    % Compute conditional covariances
    for i=1:modelPD.nbStates
        SigmaOutTmp = pSigma(out,out,i) - pSigma(out,in,i)/pSigma(in,in,i)*pSigma(in,out,i);
        expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaOutTmp + uOut(:,i,t)*uOut(:,i,t)');
    end
    expSigma(:,:,t) = expSigma(:,:,t) - uhat(:,t)*uhat(:,t)';
end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots in Cartesian space
figure('position',[10 10 1800 600],'color',[1 1 1]);
clrmap = lines(nbSamples);
% Plot demonstrations of velocity manipulability ellipsoids
subplot(1,3,1); hold on;
title('\fontsize{12}Demonstrations: 2D Cartesian trajectories and manipulability ellipsoids');
for n=1:nbSamples
    for t=round(linspace(1,nbData,15))
        plotGMM([s(n).Data(1,t);s(n).Data(2,t)], 1E-1*X(2:3,2:3,t+(n-1)*nbData), clrmap(n,:), .4); % Scaled matrix!
    end
end
for n=1:nbSamples % Plots 2D Cartesian trajectories
    plot(s(n).Data(1,:),s(n).Data(2,:), 'color', [0.5 0.5 0.5], 'Linewidth', 2); % Scaled matrix!
end
axis equal;
set(gca, 'FontSize', 20)
xlabel('$x_1$', 'Fontsize', 28, 'Interpreter', 'Latex');
ylabel('$x_2$', 'Fontsize', 28, 'Interpreter', 'Latex');

% Plot GMM
subplot(1,3,2); hold on;
title('\fontsize{12}Manipulability GMM means');
clrmap = lines(modelPD.nbStates);
sc = 1/modelPD.dt;
for n=1:nbSamples
	for t=round(linspace(1,nbData,15))
		plotGMM([s(n).Data(1,t);s(n).Data(2,t)], 1E-1*X(2:3,2:3,t+(n-1)*nbData), [.6 .6 .6], .1); % Scaled matrix!
	end
end
for i=1:modelPD.nbStates % Plotting GMM of man. ellipsoids
    xtmp = GMR(modelKin, modelPD.MuMan(in,i), 1, 2:modelKin.nbVar);
	plotGMM([xtmp], 1E-1*vec2symmat(modelPD.MuMan(out,i)), clrmap(i,:), .3);
end
axis equal;
set(gca, 'FontSize', 20)
xlabel('$x_1$', 'Fontsize', 28, 'Interpreter', 'Latex');
ylabel('$x_2$', 'Fontsize', 28, 'Interpreter', 'Latex');

% Plot desired reproduction
subplot(1,3,3); hold on;
title('\fontsize{12}Desired reproduction');
for t=1:5:nbData % Plotting estimated man. ellipsoid from GMR
    plotGMM(xd(:,t), 5E-2*vec2symmat(xhat(:,t)), [0.2 0.8 0.2], .5, '-.', 2, 1); % Scaled matrix!
end
axis equal;
set(gca, 'FontSize', 20)
xlabel('$x_1$', 'Fontsize', 28, 'Interpreter', 'Latex');
ylabel('$x_2$', 'Fontsize', 28, 'Interpreter', 'Latex');

%% Time-based plots
% Plot demonstrations of velocity manipulability ellipsoids over time
figure('position',[10 10 1200 550],'color',[1 1 1]);
clrmap = lines(nbSamples);
subplot(2,2,1); hold on;
title('\fontsize{12}Demonstrated manipulability');
for n=1:nbSamples
    for t=round(linspace(1,nbData,15))
        plotGMM([t;0], X(2:3,2:3,t+(n-1)*nbData), clrmap(n,:), .4);
    end
end
xaxis(-10, nbData+10);
set(gca, 'FontSize', 16)
ylabel('{\boldmath$M$}', 'Fontsize', 24, 'Interpreter', 'Latex');

subplot(2,2,3); hold on;
title('\fontsize{12}Demonstrated manipulability and GMM centers');
clrmap = lines(modelPD.nbStates);
sc = 1/modelPD.dt;
for t=1:size(X,3) % Plotting man. ellipsoids from demonstration data
    plotGMM([X(in,in,t)*sc; 0], X(outMat,outMat,t), [.6 .6 .6], .1);
end
for i=1:modelPD.nbStates % Plotting GMM of man. ellipsoids
    plotGMM([modelPD.MuMan(in,i)*sc; 0], vec2symmat(modelPD.MuMan(out,i)), clrmap(i,:), .3);
end
xaxis(xIn(1)*sc, xIn(end)*sc);
set(gca, 'FontSize', 16)
xlabel('$t$', 'Fontsize', 24, 'Interpreter', 'Latex');
ylabel('{\boldmath$M$}', 'Fontsize', 24, 'Interpreter', 'Latex');

subplot(2,2,2); hold on;
title('\fontsize{12}Desired manipulability profile (GMR)');
for t=1:7:nbData % Plotting estimated man. ellipsoid from GMR
    plotGMM([xIn(1,t)*sc; 0], vec2symmat(xhat(:,t)), [0.2 0.8 0.2], .5, '-.', 2, 1);
end
axis([xIn(1)*sc, xIn(end)*sc, -15, 15]);
set(gca, 'FontSize', 16)
ylabel('{\boldmath$M_d$}', 'Fontsize', 24, 'Interpreter', 'Latex');

subplot(2,2,4); hold on;  % Plotting states influence during GMR estimation
title('\fontsize{12}Influence of GMM components');
for i=1:modelPD.nbStates
    plot(xIn, H(i,:),'linewidth',2,'color',clrmap(i,:));
end
axis([xIn(1), xIn(end), 0, 1.02]);
set(gca, 'FontSize', 16)
xlabel('$t$', 'Fontsize', 24, 'Interpreter', 'Latex');
ylabel('$h_k$', 'Fontsize', 24, 'Interpreter', 'Latex');

%% Plots in the SPD space
figure('position',[10 10 1800 500],'color',[1 1 1]);
clrmap = lines(nbSamples);
% Plot demonstrations of velocity manipulability ellipsoids in SPD space
subplot(1,3,1); hold on;
title('\fontsize{12}Demonstrations: manipulability ellipsoids');
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot demonstrations
for n=1:nbSamples
	for t=1:nbData
        plot3(x(2, t+(n-1)*nbData), x(3, t+(n-1)*nbData), ... 
            x(4, t+(n-1)*nbData)/sqrt(2), '.', 'Markersize', 12, 'color', clrmap(n,:));
    end
end

% Plot demonstrated manipulability and GMM components in SPD space
subplot(1,3,2); hold on;
title('\fontsize{12}Manipulability GMM in SPD space');
clrmap = lines(modelPD.nbStates);
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot samples
for n=1:nbSamples
	for t=1:nbData
        plot3(x(2, t+(n-1)*nbData), x(3, t+(n-1)*nbData), x(4, t+(n-1)*nbData)/sqrt(2), '.', 'Markersize', 10, 'color', [.6 .6 .6]);
    end
end
for i=1:modelPD.nbStates % Plotting GMM of man. ellipsoids
    mu = modelPD.MuMan(out,i);
    mu(3) = mu(3)/sqrt(2); % rescale for plots
    sigma = modelPD.Sigma(out,out,i);
    sigma(3,:) = sigma(3,:)./sqrt(2); % rescale for plots
    sigma(:,3) = sigma(:,3)./sqrt(2); % rescale for plots
    sigma = sigma + 5*eye(3); % for better visualisaton of the last covariance (really thin along one axis)
	plotGMM3D(mu, sigma, clrmap(i,:), .6);
end

subplot(1,3,3); hold on;
title('\fontsize{12}Desired reproduction');
r = 200;
phi = 0:0.1:2*pi+0.1;
xax = [zeros(size(phi)); r.*ones(size(phi))];
yax = [zeros(size(phi));r.*sin(phi)];
zax = [zeros(size(phi));r/sqrt(2).*cos(phi)]; 
% Cone
h = mesh(xax,yax,zax,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(2,:),yax(2,:),zax(2,:),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,63),yax(:,63),zax(:,63),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(xax(:,40),yax(:,40),zax(:,40),'linewidth',3,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
% Draw axis
plot3([0,250],[0,0],[0,0],'k','linewidth',.5)
plot3([0,0],[0,250],[0,0],'k','linewidth',.5)
plot3([0,0],[0,0],[0,150],'k','linewidth',.5)
% Add text for axis
text(280,-40,0,'$\mathbf{M}_{11}$','FontSize',20,'Interpreter','latex')
text(15,0,120,'$\mathbf{M}_{12}$','FontSize',20,'Interpreter','latex')
text(5,220,-15,'$\mathbf{M}_{22}$','FontSize',20,'Interpreter','latex')
% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
view(70,12);
% Plot
plot3(xhat(1, :), xhat(2, :), xhat(3, :)/sqrt(2), '.', 'Markersize', 12, 'color', [0.2 0.8 0.2]);
end