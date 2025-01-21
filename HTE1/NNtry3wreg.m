%%
clear ; close all; clc

%% loading and setting up the data
data = readtable('drive_diagnosis_NN.csv');
X = data{3:end, 1:end-1}; % the first two rows were excluded and the last column since it has the classes
y = data{3:end, end}; % first two rows of the classes column were also excluded because of non numerical values

%% to ensure class labels are continuous and start from 1
[unique_classes, ~, y] = unique(y);  % this gives class vector in a non repetitive manner
num_classes = length(unique_classes);

%% feature normalization
X = (X - mean(X)) ./ std(X);

%% one-hot encoding for y
Y = zeros(length(y), num_classes);
for i = 1:length(y)
    Y(i, y(i)) = 1;
end

%% splitting the data into training and test sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.3); % 70% and 30%
idx = cv.test;
X_train = X(~idx,:);
Y_train = Y(~idx,:);
X_test = X(idx,:);
Y_test = Y(idx,:);

%% initializing network parameters
input_layer_size = size(X_train, 2);
hidden_layer_size = 2; 
output_layer_size = size(Y_train, 2);
initial_Theta1 = rand(hidden_layer_size, 1 + input_layer_size) * 2 * 0.12 - 0.12;  % here epsilon_init = 0.12
initial_Theta2 = rand(output_layer_size, 1 + hidden_layer_size) * 2 * 0.12 - 0.12;

% training parameters
alpha = 1; % learning rate
num_iters = 10000; % number of iterations
lambda = 1; % regularization parameter

% adding ones to the traing and testing feature matrices
X_train = [ones(size(X_train, 1), 1) X_train];
X_test = [ones(size(X_test, 1), 1) X_test];


Theta1_reg = initial_Theta1(:, 2:end); % for excluding the bias term weights from regularization
Theta2_reg = initial_Theta2(:, 2:end);

%% initializing the gradients
Theta1_grad = zeros(size(initial_Theta1));
Theta2_grad = zeros(size(initial_Theta2));

%% trainging the neural network
for iter = 1:num_iters

    % forward propagation
    z2 = X_train * initial_Theta1';
    a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
    z3 = a2 * initial_Theta2';
    a3 = sigmoid(z3);
    
    % regularized cost function 
    regularization_term = (lambda / (2 * size(X_train, 1))) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));
    J = (1/size(X_train, 1)) * sum(sum(-Y_train .* log(a3) - (1 - Y_train) .* log(1 - a3))) + regularization_term;
    
    % backpropagation
    delta3 = a3 - Y_train;
    delta2 = (delta3 * initial_Theta2) .* [ones(size(z2, 1), 1) sigmoidGradient(z2)];
    delta2 = delta2(:,2:end); % Taking off the bias row
    
    % accumulate gradients
    Theta1_grad = (delta2' * X_train) / size(X_train, 1);
    Theta2_grad = (delta3' * a2) / size(X_train, 1);

    
    % regularized gradients
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / size(X_train, 1)) * Theta1_reg;
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / size(X_train, 1)) * Theta2_reg;
    
    % gradient descent parameter update
    initial_Theta1 = initial_Theta1 - alpha * Theta1_grad;
    initial_Theta2 = initial_Theta2 - alpha * Theta2_grad;
    
    % printing the cost every 1000 iterations
    if mod(iter, 1000) == 0
        fprintf('Iteration: %d | Cost: %f\n', iter, J);
    end
end

%% calculating prediction and accuracy
% forward propagation on the test set
z2_test = X_test * initial_Theta1';
a2_test = [ones(size(z2_test, 1), 1) sigmoid(z2_test)];
z3_test = a2_test * initial_Theta2';
a3_test = sigmoid(z3_test);

regularization_term_test = (lambda / (2 * size(X_test, 1))) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));
test_cost = (1/size(X_test, 1)) * sum(sum(-Y_test .* log(a3_test) - (1 - Y_test) .* log(1 - a3_test))) + regularization_term_test;

[~, predictions] = max(a3_test, [], 2);
[~, actuals] = max(Y_test, [], 2);
accuracy = mean(double(predictions == actuals)) * 100;
fprintf('Test Set Accuracy: %f\n', accuracy);

%% 
function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end
