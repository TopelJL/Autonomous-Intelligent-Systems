% ---------------------------------------------------------------------
%               Autonomous & Intelligent Systems Labratory
%                     University of Central Florida
%                            Jaxon Topel
% ---------------------------------------------------------------------
%   Description: Code is an example of a linear kalman filter
%   problem from Colorado Boulder ECE 5550: Applied Kalman filtering
&   by Dr. Gregory L. Plett.
% ---------------------------------------------------------------------

% Initialize simulation variables
% Process-noise random input.
SigmaW = 1;

% Sensor-noise random input.
SigmaV = 1;

% Plant Matrices.
A = 1;
B = 1;
C = 1;
D = 0;

% Establish number of iterations.
maxIter = 40;

% Initialize true system.
xtrue = 0;

% Initialize KF initial state.
xhat = 0;

% Initialize kalman filter covariance.
SigmaX = 0;

% Initial driving input.
u = 0;

% Storage for variables we want to plot.
xstore = zeros(length(xtrue), maxIter+1);
xstore(:,1) = xtrue;

xhatstore = zeros(length(xhat), maxIter);

SigmaXstore = zeros(length(xhat)^2, maxIter);

% Loop through all iterations (40)
for k = 1 : maxIter
    % KF Step 1a: State estimate time update.
    % In this step we are predicting the present state given only past
    % measurements. Use prior value of u.
    xhat = A*xhat + B*u;
    
    % KF Step 1b: Error Covariance time update.
    % A*SigmaX*A' term tells us we get more certain of the state estimate as time goes on. as the covariance gets smaller.
    % Calculating the covariance of the prediction error.
    SigmaX = A*SigmaX*A' + SigmaW;

    % Implied operation of system in background, with input signal u, and
    % output signal z.

    % Random input signal.
    u  = 0.5*randn(1) + cos(k/pi);

    % Represents our random process input noise.
    w = chol(SigmaW)'*randn(length(xtrue));

    % Represents our random sensor input noise.
    v = chol(SigmaV)'*randn(length(C*xtrue));

    % Z is based on present x and u.
    ztrue = C*xtrue + D*u + v;

    % Future x is based on present u.
    xtrue = A*xtrue + B*u + w;
    
    % KF Step 1c: Predict System output.
    % % Best guess of the system output, given only past measurements.
    zhat = C*xhat + D*u;

    % KF Step 2a: Estimator (Kalman) gain matrix.
    % This is the part of Kalman filtering that distinguishes from other estimation methods.
    % We calculate Covariance matrix so we can update L.
    % SigmaX*C' indicates relative need for correction and how well states are coupled to the measurements.
    % SigmaX*C' Tells us about the state uncertainty.
    % Larger SigmaX*C' means state is very uncertain, and would benefit from large update.
    % Smaller SigmaX*C' means state is certain.
    % C' gives the coupling between the state and the output. Zero entries means no influence, large entries means that the state has high influence to output prediction error.
    % C*SigmaX*C'  indicates how the error in the state contributes to the error in the output estimnate.
    % SigmaV indicates uncertainty in the sensore reading due to noise.
    L = SigmaX*C'/(C*SigmaX*C' + SigmaV);

    % KF Step 2b: State estimate measurement update.
    % Compute a state estimate by updating the priori estimate using the
    % estimator gai and the output prediction error.
    % Zhat is what we expect the measurement to be based on our state estimate.
    xhat = xhat + L*(ztrue - zhat);

    % KF Step 2c: Error covariance measurement update.
    % This measurement update will decrease our uncertainty in the state
    % estimate.
    SigmaX = SigmaX - L*C*SigmaX;

    % Store info for plotting and evaluation.
    xstore(:, k+1) = xtrue;
    xhatstore(:,k) = xhat;
    SigmaXstore(:, k) = SigmaX(:);
end

% Plot the kalman filter in action.
% Estimation error bounds are +- 3 sqrt(SigmaX) for 99% assurance of
% accuracy of estimate.
figure(1); clf;
plot(0:maxIter-1,xstore(1:maxIter)','k-',0:maxIter-1,xhatstore','b--', ...
0:maxIter-1,xhatstore'+3*sqrt(SigmaXstore'),'m-.',...
0:maxIter-1,xhatstore'-3*sqrt(SigmaXstore'),'m-.'); grid;
legend('true','estimate','bounds');
title('Kalman filter in action'); xlabel('Iteration'); ylabel('State');

% Plot our error.
% Estimation error bounds are +- 3 sqrt(SigmaX) for 99% assurance of
% accuracy of estimate.
figure(2); clf;
plot(0:maxIter-1,xstore(1:maxIter)'-xhatstore','b-',...
0:maxIter-1,3*sqrt(SigmaXstore'),'m--',...
0:maxIter-1,-3*sqrt(SigmaXstore'),'m--'); grid;
legend('Error','bounds',0);
title('Error w/ bounds'); xlabel('Iteration'); ylabel('Estimation Error');
