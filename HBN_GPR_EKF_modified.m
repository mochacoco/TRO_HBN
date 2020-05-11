# GPR with EKF implementation

clear all;
close all;
clc;
load('data_modified.mat');

A = data(1:6830,1:4);
B = data(1:6830,5:8);
C = data(1:4400,9:11);

gprD = fitrgp(A(:,1:3),A(:,4),'Kernelfunction','ardsquaredexponential','OptimizeHyperparameters', 'sigma', 'sigma', std(A(1:20,4)), ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',10,'ShowPlots',1,'UseParallel',false));
gprS = fitrgp(B(:,1:3),B(:,4),'Kernelfunction','ardsquaredexponential','OptimizeHyperparameters','sigma', 'sigma', std(B(1:20,4)), ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',10,'ShowPlots',1,'UseParallel',false));
State = zeros(4400,4);
xkk = [C(1,1) C(1,2) 0 0];
State(1,:) = xkk;
Pkk = zeros(4,4);
e = 1e-5;
for i=2:1:4400
    disp(i);
    zk = C(i,3);
    uk = C(i,1);
    [x2kk1,w] = predict(gprD, [uk xkk(1) xkk(2)]);
    [x4kk1,v] = predict(gprS, [xkk(2) xkk(3) xkk(4)]);
    xkk_1 = [uk x2kk1 xkk(2) x4kk1]';
    F1 = (predict(gprD, [uk xkk(1)+e xkk(2)]) - x2kk1)./e;
    F2 = (predict(gprD, [uk xkk(1) xkk(2)+e]) - x2kk1)./e;
    F3 = (predict(gprS, [xkk(2)+e xkk(3) xkk(4)]) - x4kk1)./e;
    F4 = (predict(gprS, [xkk(2) xkk(3)+e xkk(4)]) - x4kk1)./e;
    F5 = (predict(gprS, [xkk(2) xkk(3) xkk(4)+e]) - x4kk1)./e;
    Hk = [0 0 0 1];
    Fk_1 = [0 0 0 0; F1 F2 0 0; 0 1 0 0; 0 F3 F4 F5];
    Qk_1 = [0 0 0 0; 0 w^2 0 0; 0 0 0 0; 0 0 0 0];
    Rk = v^2;
    Pkk_1 = Fk_1*Pkk*Fk_1' + Qk_1;
    
    ykhat = zk - x4kk1;
    Sk = Hk*Pkk_1*Hk' + Rk;
    Kk = Pkk_1 * Hk' * inv(Sk);
    
    xkk = xkk_1 + Kk*ykhat;
    Pkk = (eye(4) - Kk*Hk)*Pkk_1;
    
    State(i,:) = xkk;
end
plot(C(1:4400,2));
hold on;
plot(State(1:4400,2));
