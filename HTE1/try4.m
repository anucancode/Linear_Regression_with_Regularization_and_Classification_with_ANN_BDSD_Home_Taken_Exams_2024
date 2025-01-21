%%
clear all; 
close all; 
clc;
%% loading the dataset
data = readtable('energy_efficiency_data_heating_load.csv');

%% extracting features and target variable
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));
n = size(data, 2) - 1; % Number of features
m = length(y); % Number of observations

%% normalization 
normalized_X = (X - mean(X)) ./ std(X);
normalized_data = [normalized_X, y];

%% splitting data into train, validation and test sets
total_count = height(normalized_data);

% split ratios for train, validation, and test sets
train_ratio = 0.6;
validation_ratio = 0.2;
test_ratio = 1 - train_ratio - validation_ratio; 

% indices for splitting
train_idx = floor(total_count * train_ratio);
validation_idx = train_idx + floor(total_count * validation_ratio);

% splitting
train_data = normalized_data(1:train_idx, :);
validation_data = normalized_data((train_idx + 1):validation_idx, :);
test_data = normalized_data((validation_idx + 1):end, :);

%% preparing and visualizing the training set
X_train = train_data(:, 1:end-1); 
y_train = train_data(:, end); 


figure
scatter(X_train(:,1), y_train)
title('Visualizing the training set (Normalized) for Feature 1')
xlabel('Feature 1')
ylabel('Heating load')
grid on

%% preparing the data for linear regression
m = size(X_train, 1); % no. of observations in the training set
X_train = [ones(m,1), X_train]; % intercept term

% initial theta 
theta = zeros(n+1, 1);

% learning parameters
alpha = 0.01; % learning rate
lambda = 0.1; % regularization parameter
iterations = 1000; 

% placeholder for the cost function history
J_history = zeros(iterations, 1);

%% training the model with regularized gradient descent
for iter = 1:iterations
    h = X_train * theta;
    error = h - y_train;
    % regularized gradient descent update formula
    theta = theta - (alpha/m) * (X_train' * error + lambda * theta);
    % regularized cost function
    J_history(iter) = (1/(2*m)) * sum(error .^ 2) + (lambda/(2*m)) * sum(theta(2:end) .^ 2);
end
%% plotting J_history
figure;
plot(1:iterations, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost');
title('Cost Function History');
grid on
%% visualizing the training 
figure;
scatter(X_train(:,2), y_train); % with first feature 
hold on;
plot(X_train(:,2), h, '-r', 'LineWidth', 2);
xlabel('Feature 1');
ylabel('Heating Load');
title('Training Data with Regression Line');
legend('Training Data', 'Regression Line','Location','northwest');
grid on
hold off;
%% validation
X_val = [ones(size(validation_data, 1), 1), validation_data(:, 1:end-1)]; % also adding intercept term to the validation set features
y_val = validation_data(:, end);
h_val = X_val * theta; % predictions on the validation set

% mean squared error (MSE) for the validation set
mse_val = mean((h_val - y_val).^2);
fprintf('Mean Squared Error on Validation Set: %f\n', mse_val);

% visualizing validation performance
figure;
scatter(y_val, h_val);
hold on;
max_val = max(max(y_val), max(h_val)); % scaling the plot correctly
plot([0 max_val], [0 max_val], '-r', 'LineWidth', 2); % line for perfect predictions
xlabel('Actual Heating Load');
ylabel('Predicted Heating Load');
legend('Predictions', 'Ideal Prediction','Location','northwest');
hold off;
grid on;
%% testing

X_test = [ones(size(test_data, 1), 1), test_data(:, 1:end-1)]; % also adding intercept term to the test set features
y_test = test_data(:, end); 

h_test = X_test * theta; % predictions on the test set

% mean squared error (MSE) for the test set
mse_test = mean((h_test - y_test).^2);
fprintf('Mean Squared Error on Test Set: %f\n', mse_test);

% visualizing test performance
figure;
scatter(y_test, h_test); 
hold on;
max_val_test = max(max(y_test), max(h_test)); % scaling the plot correctly
plot([0 max_val_test], [0 max_val_test], '-r', 'LineWidth', 2); % line for perfect predictions
xlabel('Actual Heating Load');
ylabel('Predicted Heating Load');
legend('Test Set Predictions', 'Ideal Prediction','Location','northwest');
hold off;
grid on;
