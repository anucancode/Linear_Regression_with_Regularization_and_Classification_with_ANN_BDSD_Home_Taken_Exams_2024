%%
clear all; 
close all; 
clc;
%% loading the dataset
data = readtable('drive_diagnosis_NN.csv');
%% setting up the data
X = data{3:end, 1:end-1}; % Assuming the last column is the class label
y = data{3:end, end};

%% normalize features
X_norm = (X - mean(X)) ./ std(X);
%% splitting into train and test sets
cv = cvpartition(size(data, 1), 'HoldOut', 0.3); % 70-30 split
idxTrain = training(cv);
idxTest = test(cv);

Xtrain = X_norm(idxTrain);
Ytrain = y(idxTrain);

Xtest = X_norm(idxTest);
Ytest = y(idxTest);

%%
% Assuming input features size is 'n' and there are 'm' classes
n = size(Xtrain, 2); % number of features
m = unique(Ytrain); % number of unique classes
inputSize = size(Xtrain,2);
numClasses = length(m); % assuming your classes are 1, 2, ..., numClasses
hiddenSize = 2;
%%
%[inputSize, m] = size(X); % m is number of examples
%[numClasses, ~] = size(Y); % Assuming Y is one-hot encoded
    
% Initialize parameters
%inputSize = 48; % Example, replace with the actual number of features
%hiddenSize = 2;
%numClasses = 7; % Replace with the actual number of classes
[W1, b1, W2, b2] = initializeParameters(inputSize, hiddenSize, numClasses);


numIterations = 100;

for i = 1:numIterations
        % Forward propagation
        [A1, A2] = forwardPropagation(X, W1, b1, W2, b2);
        
        % Compute cost
        cost = computeCost(A2, y, m);
        
        % Backpropagation
        [dW1, db1, dW2, db2] = backPropagation(X, y, A1, A2, W2, m);
        
        % Update parameters
        [W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate);
        
        % Print the cost every 100 iterations
        if mod(i, 100) == 0
            fprintf('Cost after iteration %i: %f\n',i, cost);
        end
    end
    
    model.W1 = W1; model.b1 = b1; model.W2 = W2; model.b2 = b2;
%%
function [W1, b1, W2, b2] = initializeParameters(inputSize, hiddenSize, numClasses)
    W1 = randn(hiddenSize, inputSize) * 0.01;
    b1 = zeros(hiddenSize, 1);
    W2 = randn(numClasses, hiddenSize) * 0.01;
    b2 = zeros(numClasses, 1);
end
%%
function [A1, A2] = forwardPropagation(X, W1, b1, W2, b2)
    % Hidden layer
    Z1 = W1 * X' + b1;
    A1 = 1 ./ (1 + exp(-Z1)); % Sigmoid function
    
    % Output layer
    Z2 = W2 * A1 + b2;
    A2 = exp(Z2) ./ sum(exp(Z2), 1); % Softmax function
end
%%
function J = computeCost(A2, y, m)
    % Cross-entropy cost
    J = -sum(sum(y' .* log(A2))) / m;
end
%%
function [dW1, db1, dW2, db2] = backPropagation(X, y, A1, A2, W2, m)
    dZ2 = A2' - y;
    dW2 = (1/m) .* dZ2 .* A1;
    db2 = (1/m) * sum(dZ2, 2);
    
    dZ1 = (W2' * dZ2) .* A1 .* (1-A1);
    dW1 = (1/m) * dZ1 * X';
    db1 = (1/m) * sum(dZ1, 2);
end
%%
function [W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)
    W1 = W1 - learningRate * dW1;
    b1 = b1 - learningRate * db1;
    W2 = W2 - learningRate * dW2;
    b2 = b2 - learningRate * db2;
end
%%
function model = trainNN(X, Y, hiddenSize, numIterations, learningRate)
    % [inputSize, m] = size(X); % m is number of examples
    % [numClasses, ~] = size(Y); % Assuming Y is one-hot encoded
    % 
    % % Initialize parameters
    % [W1, b1, W2, b2] = initializeParameters(inputSize, hiddenSize, numClasses);
    
    % for i = 1:numIterations
    %     % Forward propagation
    %     [A1, A2] = forwardPropagation(X, W1, b1, W2, b2);
    % 
    %     % Compute cost
    %     cost = computeCost(A2, Y, m);
    % 
    %     % Backpropagation
    %     [dW1, db1, dW2, db2] = backPropagation(X, Y, A1, A2, W2, m);
    % 
    %     % Update parameters
    %     [W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate);
    % 
    %     % Print the cost every 100 iterations
    %     if mod(i, 100) == 0
    %         fprintf('Cost after iteration %i: %f\n',i, cost);
    %     end
    % end
    % 
    % model.W1 = W1; model.b1 = b1; model.W2 = W2; model.b2 = b2;
end
