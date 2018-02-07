
% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

% Generate Haar feature masks
nbrHaarFeatures = 1;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

% figure(3);
% colormap gray;
% for k = 1:25
%     subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%     axis image;
%     axis off;
% end

%
% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 200;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
D = (1/(2*nbrTrainExamples))*ones(nbrHaarFeatures,2*nbrTrainExamples);

T = 20;
alpha = ones(1,T);
weakC = ones(T,2*nbrTrainExamples);
for t =1:T
    EfeatureMin = Inf;
    PfeatureMin = Inf;
    thresholdFeatureMin = Inf;
    cFeatureMin = Inf*ones(nbrHaarFeatures,2*nbrTrainExamples);
    for j=1:nbrHaarFeatures
        Pmin = Inf;
        thresholdMin = Inf;
        Emin = Inf;
        Cmin = Inf*ones(nbrHaarFeatures,2*nbrTrainExamples);
        for i=1:2*nbrTrainExamples
            threshold = xTrain(j,i);
            P = 1;
            C = WeakClassifier(threshold,P,xTrain);
            E = WeakClassifierError(C,D,yTrain);
            if E > 0.5
                P = -1;
                
                E = 1 - E;
                
            end
            if E < Emin
                Pmin = P;
                thresholdMin = threshold;
                Emin = E;
                Cmin = C;
            end
        end
        if EfeatureMin > Emin
            EfeatureMin = Emin;
            thresholdFeatureMin = thresholdMin;
            cFeatureMin = Cmin;
        end
    end
    EfeatureMin
    weakC(t,:) = cFeatureMin;
      if ~isreal((1/2)*log((1-EfeatureMin)/EfeatureMin))
        t;
    end
    alpha(1,t) = (1/2)*log((1-EfeatureMin)/EfeatureMin)
    t
    %Ej hel säker
    D = D.*exp(-alpha(1,t)*yTrain.*cFeatureMin);
    D = D./sum(D);
end

strongC = sign(alpha*weakC);


% Extract test data

nbrTestExamples = 3;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.





% Plot the error of the strong classifier as  function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.


