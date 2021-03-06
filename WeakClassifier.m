function C = WeakClassifier(T, P, X)
% WEAKCLASSIFIER Classify images using a decision stump.
% Takes a vector X of scalars obtained by applying one Haar feature to all
% training images. Classifies the examples using a decision stump with
% cut-off T and parity P. Returns a vector C of classifications for all
% examples in X.

% C = ones(size(X,1),size(X,2));

C = ( P*X < P*T)*2-1;
% 
% C(mask) = -1;


% You are not allowed to use a loop in this function.
% This is for your own benefit, since a loop will be too slow to use
% with a reasonable amount of Haar features and training images.
    
end

