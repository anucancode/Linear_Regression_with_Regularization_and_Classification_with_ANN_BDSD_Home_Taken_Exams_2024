%%
clc
clear all
clear
%%
% loading data from csv file
data = readtable('energy_efficiency_data_heating_load.csv');
%%
% Assuming the last column is the target variable (heating load) and the
% rest are independent variables (features like wall area, roof area, etc.)
X = table2array(data(:, 1:end-1)); % Convert all but the heating_load column to a matrix
y = table2array(data(:, end)); % Convert the heating_load column to a vector
% X = normalizeFeatures(X);
% Calculating the number of observations and features
m = size(X, 1); % Number of rows in X, representing the observations
n = size(X, 2); % Number of columns in X, representing the features

%%
normalize = true


if normalize
  % ASSIGNMENT: implement feature normalization. This function must return:
  % 1. The normalized feature matrix
  % 2. mu and sigma to scale inputs for new predictions
  [X, mu, sigma] = normalizeFeatures(X);
  alpha = 0.3;
  iterations = 1000;
else
  mu = zeros(1, n);
  sigma = ones(1, n);
  % ASSIGNMENT: without normalization this combination of learning rate and iterations does not converge. Find
  % values that do, and compare with normalized learning.
  alpha = 20;
  iterations = 1000;
end

%%
% Add Y-displacement term
X = [ones(m, 1) X];

% Initial theta can be anything 
theta = ones(n+1, 1);

%% lin reg
    for iter = 1:iterations
        % Hypothesis function
        h = X * theta;
        % Error
        error = h - y;
        % Gradient descent update rule
        theta = theta - (alpha/m) * (X' * error);
        % Compute cost function
        J_history(iter) = (1/(2*m)) * sum(error .^ 2);
    end
%%
% % Exclude the target variable from normalization
% features = table2array(data(:, 1:end-1));
% target = table2array(data(:, end));
% 
% % Normalize the features
% normalized_features = (features - mean(features)) ./ std(features);
% 
% % Combine back with the target
% normalized_data = [normalized_features, target];
% 
% figure()
% scatter(normalized_features, target)

%%
% Calculate the interquartile range (IQR)
% Q1 = quantile(X, 0.25);
% Q3 = quantile(X, 0.75);
% IQR = Q3 - Q1;
% 
% % Find outliers
% outlier_idx = any(X < (Q1 - 1.5 * IQR) | X > (Q3 + 1.5 * IQR), 2);
% 
% % Remove outliers
% cleaned_data = normalized_data(~outlier_idx, :)
% 
% X = cleaned_data(:, 1:end-1); % Convert all but the heating_load column to a matrix
% y = cleaned_data(:, end);
%%
figure()
scatter(X,y)
%%
% % Create the scatter plot
% scatter(wallArea, heatingLoad, 'filled');
% 
% % Add title and axis labels
% title('Wall Area vs. Heating Load');
% xlabel('Wall Area (square meters)');
% ylabel('Heating Load');
% 
% % Optional: Add grid lines to the plot
% grid on;


%% normalize function
function [X_norm, mu, sigma] = normalizeFeatures(X)
    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;
end

%%
function [theta, J_history] = mvgd(X, y, theta_0, alpha, iterations)
 % Initialize variables
    m = length(y); % Number of training examples
    theta = theta_0; % Initial theta values
    J_history = zeros(iterations, 1); % To record cost function value at each iteration
end




