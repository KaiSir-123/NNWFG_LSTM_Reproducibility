clc;
close all;

%% LSTM adaptive weighting module
% Required variables:
% err_IMU, err_LiDAR, err_RTK, N_RTK_raw, D_LiDAR_raw, t
% Input : [err_IMU, err_LiDAR, err_RTK, N_RTK, D_LiDAR]
% Output: [w_IMU, w_RTK, w_LiDAR]

requiredVars = {'err_IMU','err_LiDAR','err_RTK','N_RTK_raw','D_LiDAR_raw','t'};

for i = 1:numel(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        warning(['Variable "', requiredVars{i}, '" is missing. ', ...
                 'Please prepare it from the preprocessing stage.']);
    end
end

%% 1. Feature alignment
err_IMU   = err_IMU(:);
err_LiDAR = err_LiDAR(:);
err_RTK   = err_RTK(:);
N_RTK_raw = N_RTK_raw(:);
D_LiDAR_raw = D_LiDAR_raw(:);
t = t(:);

N_feature = min([ ...
    length(err_IMU), ...
    length(err_LiDAR), ...
    length(err_RTK), ...
    length(N_RTK_raw), ...
    length(D_LiDAR_raw), ...
    length(t) ...
]);

err_IMU   = err_IMU(1:N_feature);
err_LiDAR = err_LiDAR(1:N_feature);
err_RTK   = err_RTK(1:N_feature);
N_RTK_raw = N_RTK_raw(1:N_feature);
D_LiDAR_raw = D_LiDAR_raw(1:N_feature);
t = t(1:N_feature);

%% 2. Residual smoothing
win = 50;

err_IMU_s   = movmean(err_IMU, win);
err_LiDAR_s = movmean(err_LiDAR, win);
err_RTK_s   = movmean(err_RTK, win);

%% 3. Feature construction
feature_raw = [ ...
    err_IMU_s, ...
    err_LiDAR_s, ...
    err_RTK_s, ...
    N_RTK_raw, ...
    D_LiDAR_raw ...
];

%% 4. Normalization
train_ratio = 0.70;
train_end_idx = floor(train_ratio * N_feature);

feature_min = min(feature_raw(1:train_end_idx,:), [], 1);
feature_max = max(feature_raw(1:train_end_idx,:), [], 1);

feature_LSTM = (feature_raw - feature_min) ./ ...
               (feature_max - feature_min + eps);

feature_LSTM = max(min(feature_LSTM, 1), 0);

err_IMU_input   = feature_LSTM(:,1);
err_LiDAR_input = feature_LSTM(:,2);
err_RTK_input   = feature_LSTM(:,3);
N_RTK_input     = feature_LSTM(:,4);
D_LiDAR_input   = feature_LSTM(:,5);

%% 5. Adaptive label construction
m = 50;
lambda = 1.0;

errMat = [err_IMU_s, err_RTK_s, err_LiDAR_s];

psiLabel = zeros(N_feature, 3);

for j = 1:3
    err_j = errMat(:,j);

    errMean = movmean(err_j, [m-1 0]);

    k0 = errMean;
    k1 = 3 * errMean;

    psiLabel(:,j) = adaptiveConfidence(err_j, k0, k1, lambda);
end

qIMU   = psiLabel(:,1);
qRTK   = psiLabel(:,2) .* N_RTK_input;
qLiDAR = psiLabel(:,3) .* D_LiDAR_input;

Yraw = [qIMU, qRTK, qLiDAR];
Yraw = Yraw + 1e-6;

Ylabel = Yraw ./ sum(Yraw, 2);

%% 6. Sequence construction
T = 10;
stride = 1;

[X, Y, t_out] = buildSequenceDataset(feature_LSTM, Ylabel, t, T, stride);

numSamples = numel(X);

%% 7. Dataset split
nTrain = floor(0.70 * numSamples);
nVal   = floor(0.15 * numSamples);
nTest  = numSamples - nTrain - nVal;

XTrain = X(1:nTrain);
YTrain = Y(1:nTrain,:);

XVal = X(nTrain+1:nTrain+nVal);
YVal = Y(nTrain+1:nTrain+nVal,:);

XTest = X(nTrain+nVal+1:end);
YTest = Y(nTrain+nVal+1:end,:);

%% 8. LSTM network
layers = [
    sequenceInputLayer(5, "Name","input", "Normalization","none")

    lstmLayer(64, "OutputMode","last", "Name","lstm")

    fullyConnectedLayer(32, "Name","fc_1")
    reluLayer("Name","relu_1")
    dropoutLayer(0.2, "Name","dropout")

    fullyConnectedLayer(3, "Name","fc_2")
    softmaxLayer("Name","softmax")

    regressionLayer("Name","regression")
];

%% 9. Training options
options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-3, ...
    "MaxEpochs", 100, ...
    "MiniBatchSize", 128, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {XVal, YVal}, ...
    "ValidationPatience", 10, ...
    "Verbose", true, ...
    "Plots", "training-progress");

%% 10. Training
net = trainNetwork(XTrain, YTrain, layers, options);

%% 11. Dynamic weight prediction
YPred = predict(net, X, "MiniBatchSize", 128);

YPred = max(YPred, 0);
YPred = YPred ./ (sum(YPred, 2) + eps);

w_IMU   = YPred(:,1);
w_RTK   = YPred(:,2);
w_LiDAR = YPred(:,3);

%% 12. Test evaluation
YPredTest = predict(net, XTest, "MiniBatchSize", 128);

YPredTest = max(YPredTest, 0);
YPredTest = YPredTest ./ (sum(YPredTest, 2) + eps);

rmseWeight = sqrt(mean((YPredTest - YTest).^2, 1));
testMAE = mean(abs(YPredTest - YTest), 1);
testRMSE_mean = mean(rmseWeight);

fprintf('\n========== LSTM Dynamic Weight Prediction Results ==========\n');
fprintf('Number of test samples = %d\n', size(YTest,1));
fprintf('IMU weight RMSE   = %.6f, MAE = %.6f\n', rmseWeight(1), testMAE(1));
fprintf('RTK weight RMSE   = %.6f, MAE = %.6f\n', rmseWeight(2), testMAE(2));
fprintf('LiDAR weight RMSE = %.6f, MAE = %.6f\n', rmseWeight(3), testMAE(3));
fprintf('Mean RMSE         = %.6f\n', testRMSE_mean);

%% 13. Plot dynamic weights
figure('Color','w','Position',[200 120 950 420]);
hold on;
box on;
grid on;

plot(t_out, w_IMU,   '-',  'LineWidth',1.5);
plot(t_out, w_RTK,   '--', 'LineWidth',1.5);
plot(t_out, w_LiDAR, '-.', 'LineWidth',1.5);

xlabel('Time (s)', 'FontName','Times New Roman', 'FontSize',12);
ylabel('Dynamic factor weight', 'FontName','Times New Roman', 'FontSize',12);

title('Dynamic outputs of the LSTM-based weighting model', ...
    'FontName','Times New Roman', 'FontSize',12);

legend({'IMU weight','RTK weight','LiDAR weight'}, ...
    'FontName','Times New Roman', ...
    'Location','best');

set(gca, 'FontName','Times New Roman', ...
    'FontSize',11, ...
    'LineWidth',1, ...
    'YLim',[0 1]);

savefig('Fig_LSTM_dynamic_weight_outputs.fig');
print(gcf, 'Fig_LSTM_dynamic_weight_outputs.png', '-dpng', '-r600');

%% 14. Plot normalized input features
figure('Color','w','Position',[220 100 950 680]);

subplot(3,1,1);
plot(t, err_IMU_input, 'LineWidth',1.2);
hold on;
plot(t, err_RTK_input, 'LineWidth',1.2);
plot(t, err_LiDAR_input, 'LineWidth',1.2);
grid on;
box on;
ylabel('Normalized residual');
legend({'err_{IMU}','err_{RTK}','err_{LiDAR}'}, ...
    'FontName','Times New Roman', 'Location','best');
title('Normalized sensor residuals', 'FontName','Times New Roman');
set(gca,'FontName','Times New Roman','FontSize',11);

subplot(3,1,2);
plot(t, N_RTK_input, 'LineWidth',1.2);
grid on;
box on;
ylabel('Normalized RTK satellites');
title('Normalized RTK satellite number', 'FontName','Times New Roman');
set(gca,'FontName','Times New Roman','FontSize',11);

subplot(3,1,3);
plot(t, D_LiDAR_input, 'LineWidth',1.2);
grid on;
box on;
xlabel('Time (s)', 'FontName','Times New Roman');
ylabel('Normalized LiDAR density');
title('Normalized LiDAR point-cloud density', 'FontName','Times New Roman');
set(gca,'FontName','Times New Roman','FontSize',11);

savefig('Fig_LSTM_normalized_input_features.fig');
print(gcf, 'Fig_LSTM_normalized_input_features.png', '-dpng', '-r600');

%% 15. Save results
LSTM_dynamic_weights = [t_out, w_IMU, w_RTK, w_LiDAR];

save('NNWFG_LSTM_dynamic_weight_result.mat', ...
    'net', ...
    'feature_raw', ...
    'feature_LSTM', ...
    'feature_min', ...
    'feature_max', ...
    'Ylabel', ...
    'YPred', ...
    'LSTM_dynamic_weights', ...
    't', ...
    't_out', ...
    'err_IMU_s', ...
    'err_LiDAR_s', ...
    'err_RTK_s', ...
    'N_RTK_raw', ...
    'D_LiDAR_raw', ...
    'N_RTK_input', ...
    'D_LiDAR_input', ...
    'w_IMU', ...
    'w_RTK', ...
    'w_LiDAR', ...
    'T', ...
    'stride', ...
    'm', ...
    'lambda', ...
    'rmseWeight', ...
    'testMAE');

writematrix(LSTM_dynamic_weights, 'LSTM_dynamic_weight_outputs.csv');

%% Local functions
function psi = adaptiveConfidence(err_j, k0, k1, lambda)

psi = zeros(size(err_j));

idx1 = err_j <= k0;
idx2 = err_j > k0 & err_j <= k1;
idx3 = err_j > k1;

psi(idx1) = 1;

denominator = k1(idx2) - k0(idx2) + eps;
psi(idx2) = lambda .* (k1(idx2) - err_j(idx2)) ./ denominator;

psi(idx3) = 0;

psi = max(min(psi, 1), 0);

end

function [X, Y, t_out] = buildSequenceDataset(feature_LSTM, Ylabel, t, T, stride)

N = size(feature_LSTM, 1);
sampleIdx = T:stride:N;
numSamples = numel(sampleIdx);

X = cell(numSamples, 1);
Y = zeros(numSamples, 3);
t_out = zeros(numSamples, 1);

for i = 1:numSamples
    idxEnd = sampleIdx(i);
    idxStart = idxEnd - T + 1;

    seq = feature_LSTM(idxStart:idxEnd, :);
    X{i} = seq';

    Y(i,:) = Ylabel(idxEnd, :);
    t_out(i) = t(idxEnd);
end

end