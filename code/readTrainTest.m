function [XTrain, YTrain, XTest, YTest] = readTrainTest(difficulty)

Train = csvread(sprintf('../data/%s_TRAIN.csv', difficulty));
XTrain = Train(:, 1:1000);
YTrain = Train(:, 1001) * 2 - 1;

Test = csvread(sprintf('../data/%s_TEST.csv', difficulty));
XTest = Test(:, 1:1000);
YTest = Test(:, 1001) * 2 - 1;