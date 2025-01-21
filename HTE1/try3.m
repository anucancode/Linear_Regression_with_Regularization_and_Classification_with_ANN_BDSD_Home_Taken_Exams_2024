%%
clear all; 
close all; 
clc;
%%
data = readtable('energy_efficiency_data_heating_load.csv');

X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));
n = size(data, 2) - 1;
m = length(y);

%% normalization 
normalized_X = (X - mean(X)) ./ std(X);
normalized_data = [normalized_X, y];

%% splitting data into train, validation and test sets
total_count = height(normalized_data);

% Split ratios for train, validation, and test sets
train_ratio = 0.7;
validation_ratio = 0.15;
% the test ratio becomes the remainder (0.15 in this case)

% Calculate indices for splitting
train_idx = floor(total_count * train_ratio);
validation_idx = train_idx + floor(total_count * validation_ratio);

% Split the data
train_data = normalized_data(1:train_idx, :);
validation_data = normalized_data((train_idx + 1):validation_idx, :);
test_data = normalized_data((validation_idx + 1):end, :);

% split_ratio = 0.8;
% split_idx = floor(height(normalized_data) * split_ratio);
% 
% % splitting data 
% train_data = normalized_data(1:split_idx, :);
% validation_data = normalized_data((split_idx+1):end, :);

%% Visualizing the training set
X = train_data(:, 1:end-1); % all but the heating_load column
y = train_data(:, end); % heating_load column

figure()
scatter(X,y)
title('Visualizing the training set (Normalized)')
xlabel('Features')
ylabel('Heating load')
grid on
%% calculating the number of observations and features
n = size(train_data, 2) - 1;
m = length(y);
%%
% Y-displacement term
X = [X ones(m,1)];

% initial theta 
theta = ones(n+1, 1);
%%
alpha = 0.01; % learning rate
lambda = 0.1; % regularization parameter
iterations = 1000; 


% training the model with regularized gradient descent
[theta, J_history] = mvgd(X, y, theta, alpha, lambda, iterations);

for iter = 1:iterations
    h = X * theta;
    error = h - y;
    % regularized gradient descent update formula
    theta = theta - (alpha/m) * (X' * error + lambda * theta);
    % computing regularized cost function
    J_history(iter) = (1/(2*m)) * sum(error .^ 2) + (lambda/(2*m)) * sum(theta(2:end) .^ 2);
end

figure;
plot(1:iter, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost');
grid on

%% validation
% predictions on the validation set
X_val = [ones(size(validation_data, 1), 1), validation_data(:, 1:end-1)];  %check this matrix 
y_val = validation_data(:, end);
h_val = X_val * theta;

% calculating the mean squared error (MSE) from the validation set
mse_val = mean((h_val - y_val).^2)
%% visualizing the training on the training data
% scatter plot of the first feature against the target variable
figure;
scatter(X(:,1), y, 'filled');
hold on; 

% plotting the regression line
plot(X(:,1), h, '-r', 'LineWidth', 2);
xlabel('Feature 1');
ylabel('Target Variable');
title('Feature 1 vs. Target Variable with Regression Line');
legend('Data Points', 'Regression Line');
hold off;
%% visualizing the validation 
figure;
scatter(y_val, h_val, 'filled');
hold on;
max_val = max(max(y_val), max(h_val)); % Find the maximum value for scaling the plot correctly
plot([0 max_val], [0 max_val], '-r', 'LineWidth', 2); % Ideal line for perfect predictions
xlabel('Actual Target Value');
ylabel('Predicted Value');
title('Actual vs. Predicted on Validation Set');
legend('Predictions', 'Ideal Prediction');
hold off;
grid on

%%
function [theta, J_history] = mvgd(X, y, theta_0, alpha, lambda, iterations)
    m = length(y); % Number of training examples
    theta = theta_0;
    J_history = zeros(iterations, 1); % To record cost function value at each iteration
end 


