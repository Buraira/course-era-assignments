function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
t = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g = X*theta;
h = sigmoid(g);

temp1 = log(h);
temp2 = log(1-h);
temp3 = theta(2:t).*theta(2:t);
  
  n = sum((-y'*temp1) - ((1-y)'*temp2)); 
  J = n/m + (lambda/(2*m))*sum(temp3);
  
temp1 = sum((h - y).*X);
grad = temp1./m ;
grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*(theta(2:t))';



% =============================================================

end
