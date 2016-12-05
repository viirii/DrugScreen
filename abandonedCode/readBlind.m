function [XBlind, IDBlind] = readBlind(difficulty)
Blinded = csvread(sprintf('../data/%s_BLINDED.csv', difficulty));
IDBlind = Blinded(:, 1);
XBlind = Blinded(:, 2:1001);