# Optional / GPR with SMC

clear all;
close all;
clc;
load('data_modified.mat');

A = data(1:6830,1:4);
B = data(1:6830,5:8);
C = data(1:4400,9:11);
B(:,3:4) = B(:,3:4).*10000;
C(:,3) = C(:,3).*10000;

gprD = fitrgp(A(:,1:3),A(:,4),'Kernelfunction','ardsquaredexponential','OptimizeHyperparameters', 'sigma', 'sigma', std(A(1:20,4)), ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',10,'ShowPlots',1,'UseParallel',false));
gprS = fitrgp(B(:,1:3),B(:,4),'Kernelfunction','ardsquaredexponential','OptimizeHyperparameters','sigma', 'sigma', std(B(1:20,4)), ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',10,'ShowPlots',1,'UseParallel',false));


pn = 20;
Filter = zeros(pn,1);
State = zeros(length(C),1);
for i=1:1:pn
    Filter(i,1) = C(1,2) + rand - 0.5;
end
State(1,1) = mean(Filter);
for i=2:1:4400
    disp(i);
    Tinput = [ones(pn,2).*[C(i,1) C(i-1,1)] Filter]; %pn x 3
    %Calculate Beta
    [Tout,Tsigma] = predict(gprD, Tinput);
    
    Tout = normrnd(Tout, Tsigma);
    
    %Calculate Update
    Sinput = [Tout, Filter, ones(pn,1).*C(i-1,3)];
    [Sout, Ssigma] = predict(gprS, Sinput);
    
    Sprob = normpdf(C(i,3), Sout, Ssigma);
    Sprob = Sprob./sum(Sprob);
    Filter = Standardised_Resample(Sprob, Tout, pn);
    State(i,1) = mean(Filter);
end

plot(C(1:4400,2));
hold on;
plot(State(1:4400));


function indices = Standardised_Resample(Sprob, Filter, pn)
    inx = 0;
    indices = zeros(pn,1);
    INX = zeros(pn,1);
    for i=1:1:pn
        inx = inx + Sprob(i,1);
        INX(i,1) = inx;
    end
    for i=1:1:pn
        pro = rand;
        for j=1:1:pn
            if pro <= INX(j,1)
                break;
            end
        end
        indices(i,1) = Filter(j,1);
    end
end
