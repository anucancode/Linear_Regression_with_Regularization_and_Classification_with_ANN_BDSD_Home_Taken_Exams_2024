%%
clc
clear all
clear
%%
% loading data from csv file
data = readtable('energy_efficiency_data_heating_load.csv');

%% normalization

% Exclude the target variable from normalization
X = table2array(data(:, 1:end)); 
y = table2array(data(:, end));

% Normalize the features
normalized_X = (X - mean(X)) ./ std(X);

% Combine back with the target
normalized_data = [normalized_X, y];
%% splitting data in train and test sets
% Define the split ratio
split_ratio = 0.8;
split_idx = floor(height(normalized_data) * split_ratio);

% Split the data
train_data = normalized_data(1:split_idx, :);
validation_data = normalized_data((split_idx+1):end, :);

%%
X = train_data(:, 2:end); % Convert all but the heating_load column to a matrix
y = train_data(:, 1); % Convert the heating_load column to a vector 
% Calculating the number of observations and features
n = size(train_data, 2) - 1;
m = length(y);
%m = size(X, 1); % Number of rows in X, representing the observations
%n = size(X, 2); % Number of columns in X, representing the features

%%
alpha = 0.01; % Learning rate
lambda = 0.1; % Regularization parameter
iterations = 1000; % Number of iterations

% Initialize theta
theta = (ones(n, 1));

% Train the model with regularized gradient descent
%[theta, J_history] = mvgd(X, y, theta, alpha, lambda, iterations);

for iter = 1:iterations
        h = X * theta;
        error = h - y;
        % Regularized gradient descent update rule
        theta = theta - (alpha/m) * (X' * error + lambda * theta);
        % Compute regularized cost function
        J_history(iter) = (1/(2*m)) * sum(error .^ 2) + (lambda/(2*m)) * sum(theta(2:end) .^ 2);
    end

%% validation
% Predictions on the validation set
X_val = [ones(size(validation_data, 1), 1), validation_data(:, 1:end-1)];
y_val = validation_data(:, end);
h_val = X_val * theta;

% Calculate the mean squared error (MSE) on the validation set
mse_val = mean((h_val - y_val).^2);

%%
function [theta, J_history] = mvgd(X, y, theta_0, alpha, lambda, iterations)
    m = length(y); % Number of training examples
    theta = theta_0;
    J_history = zeros(iterations, 1); % To record cost function value at each iteration
end 
