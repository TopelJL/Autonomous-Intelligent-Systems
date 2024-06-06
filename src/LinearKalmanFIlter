% ---------------------------------------------------------------------
%               Autonomous & Intelligent Systems Labratory
%                     University of Central Florida
%                            Jaxon Topel
% ---------------------------------------------------------------------
%   Description: Code is an example of a linear kalman filter
%   problem from Colorado Boulder ECE 5550: Applied Kalman filtering
% ---------------------------------------------------------------------


% Initialize simulation variables
SigmaW = 1; SigmaV = 1; % Process and sensor noise covariance

A = 1; B = 1; C = 1; D = 0; % Plant definition matrices

maxIter = 40;

xtrue = 0; xhat = 0; % Initialize true system and KF initial state

SigmaX = 0; % Initialize Kalman filter covariance

u = 0; % Unknown initial driving input: assume zero

% Reserve storage for variables we want to plot/evaluate
xstore = zeros(length(xtrue),maxIter+1); xstore(:,1) = xtrue;
xhatstore = zeros(length(xhat),maxIter);
SigmaXstore = zeros(length(xhat)^2,maxIter);

for k = 1:maxIter,
    % KF Step 1a: State estimate time update
    xhat = A*xhat + B*u; % use prior value of "u"
    
    % KF Step 1b: Error covariance time update
    SigmaX = A*SigmaX*A' + SigmaW;
    
    % [Implied operation of system in background, with
    % input signal u, and output signal z]
    u = 0.5*randn(1) + cos(k/pi); % for example...
    w = chol(SigmaW)'*randn(length(xtrue));
    v = chol(SigmaV)'*randn(length(C*xtrue));
    
    ztrue = C*xtrue + D*u + v; % z is based on present x and u
    xtrue = A*xtrue + B*u + w; % future x is based on present u
    % KF Step 1c: Estimate system output
    zhat = C*xhat + D*u;

    % KF Step 2a: Compute Kalman gain matrix
    L = SigmaX*C'/(C*SigmaX*C' + SigmaV);
    % KF Step 2b: State estimate measurement update
    xhat = xhat + L*(ztrue - zhat);
    % KF Step 2c: Error covariance measurement update
    SigmaX = SigmaX - L*C*SigmaX;
    % [Store information for evaluation/plotting purposes]
    xstore(:,k+1) = xtrue; xhatstore(:,k) = xhat;
    SigmaXstore(:,k) = SigmaX(:);
end

figure(1); clf;
plot(0:maxIter-1,xstore(1:maxIter)','k-',0:maxIter-1,xhatstore','b--', ...
0:maxIter-1,xhatstore'+3*sqrt(SigmaXstore'),'m-.',...
0:maxIter-1,xhatstore'-3*sqrt(SigmaXstore'),'m-.'); grid;
legend('true','estimate','bounds');
title('Kalman filter in action'); xlabel('Iteration'); ylabel('State');
figure(2); clf;
plot(0:maxIter-1,xstore(1:maxIter)'-xhatstore','b-',...
0:maxIter-1,3*sqrt(SigmaXstore'),'m--',...
0:maxIter-1,-3*sqrt(SigmaXstore'),'m--'); grid;
legend('Error','bounds',0);
title('Error w/ bounds'); xlabel('Iteration'); ylabel('Estimation Error');
