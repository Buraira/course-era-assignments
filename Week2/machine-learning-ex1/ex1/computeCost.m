function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

%data = load('ex1data1.txt');
% Initialize some useful values
%X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples


%theta = zeros(2,1);
theta_j = 0;
alpha = 0.01;

X_temp = X(:,2);

n = 0;
% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


for i=1:m
  H_theta_X = theta(1,:) + (theta(2,:) * X_temp(i)) ; 
  n = n + (H_theta_X - y(i))^2;
end;


J = n/(2 * m);


% =========================================================================

end
