clear
close all
clc
format long

label = {'Car','Cyclist','Pedestrian'};

%% Train
dir_ = '...\Code\Logit_Data\Train';
load([dir_,'\Car_TP.mat'])
load([dir_,'\Cyc_TP.mat'])
load([dir_,'\Ped_TP.mat'])

nbins = 22;
lambda = 0.0055; % smoothing
%% Scores before sigmoid function: train dataset
classes_car = zeros(size(Car_TP_raw_sc,1),1);
classes_cyc = ones(size(Cyc_TP_raw_sc,1),1);
classes_ped = 2*ones(size(Ped_TP_raw_sc,1),1);
classes_det = [classes_car;classes_cyc;classes_ped];

raw_sc = [Car_TP_raw_sc;Cyc_TP_raw_sc;Ped_TP_raw_sc];
score_before_sigmoid = [raw_sc classes_det];

c = unique(classes_det);
nc = length(label); %3
ctr1 = score_before_sigmoid(:,4) == c(1); ctr2 = score_before_sigmoid(:,4) == c(2); ctr3 = score_before_sigmoid(:,4) == c(3);

figure(1);
set(gcf,'color','w');grid
cla; grid; title('TP-Train'); hold on
hc1 = histogram(score_before_sigmoid(ctr1,1),nbins,'Normalization','probability'); %class=c1 / y=c1
hc2 = histogram(score_before_sigmoid(ctr2,2),nbins,'Normalization','probability'); %class=c1 / y=c2
hc3 = histogram(score_before_sigmoid(ctr3,3),nbins,'Normalization','probability'); %class=c1 / y=c3
legend('Car','Cyc','Ped')
grid

Valores1 = hc1.Values;
BinEdgesLow1 = hc1.BinEdges(1:nbins);
BinEdgesHigh1 = hc1.BinEdges(2:nbins+1);

Valores2 = hc2.Values;
BinEdgesLow2 = hc2.BinEdges(1:nbins);
BinEdgesHigh2 = hc2.BinEdges(2:nbins+1);

Valores3 = hc3.Values;
BinEdgesLow3 = hc3.BinEdges(1:nbins);
BinEdgesHigh3 = hc3.BinEdges(2:nbins+1);

Valores = [Valores1;Valores2;Valores3];
BinEdgesLow = [BinEdgesLow1;BinEdgesLow2;BinEdgesLow3];
BinEdgesHigh = [BinEdgesHigh1;BinEdgesHigh2;BinEdgesHigh3];

%% Test
dir_test = '...\Code\Logit_Data\Test';

load([dir_test,'\Car_TP.mat'])
load([dir_test,'\Cyc_TP.mat'])
load([dir_test,'\Ped_TP.mat'])

%% TP Test: scores before sigmoid function
ClS_Test_TP = [Car_TP_raw_sc;Cyc_TP_raw_sc;Ped_TP_raw_sc];

Y = ClS_Test_TP;
n = size(Y,1);
nc = length(label);
P = zeros(n,nc);
for k=1:n % objects
    for cla=1:nc % classes
        for i=1:size(Valores,2)
            if (BinEdgesLow(cla,i) <= Y(k,cla)) && (Y(k,cla) < BinEdgesHigh(cla,i))
                P(k,cla) = Valores(cla,i);
            end
        end
    end
end
ML_TP = [];

ObjnessCar_TP = Car_TP_ObjSc;
ObjnessCyc_TP = Cyc_TP_ObjSc;
ObjnessPed_TP = Ped_TP_ObjSc;

ML_TP = P+lambda;
ML_TP = ML_TP./sum(ML_TP,2);
ML_TP(1:size(ObjnessCar_TP,1),1) = ML_TP(1:size(ObjnessCar_TP,1),1).*ObjnessCar_TP;
ML_TP(size(ObjnessCar_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1),2) = ML_TP(size(ObjnessCar_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1),2).*ObjnessCyc_TP;
ML_TP(size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+size(ObjnessPed_TP,1),3) = ML_TP(size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+size(ObjnessPed_TP,1),3).*ObjnessPed_TP;

figure2 = figure('Color',[1 1 1]);

subplot1 = subplot(1,3,1,'Parent',figure2);
hold(subplot1,'on');
histogram(ML_TP(1:size(ObjnessCar_TP,1),1),'Parent',subplot1,'Normalization','probability','NumBins',nbins);
title({'ML-TP-Car'});
box(subplot1,'on');
grid(subplot1,'on');
hold(subplot1,'off');

subplot2 = subplot(1,3,2,'Parent',figure2);
hold(subplot2,'on');
histogram(ML_TP(size(ObjnessCar_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1),2),'Parent',subplot2,'Normalization','probability','NumBins',nbins);
title({'ML-TP-Cyc'});
box(subplot2,'on');
grid(subplot2,'on');
hold(subplot2,'off');

subplot3 = subplot(1,3,3,'Parent',figure2);
hold(subplot3,'on');
histogram(ML_TP(size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+1:size(ObjnessCar_TP,1)+size(ObjnessCyc_TP,1)+size(ObjnessPed_TP,1),3),'Parent',subplot3,'Normalization','probability','NumBins',nbins);
title({'ML-TP-Ped'});
box(subplot3,'on');
grid(subplot3,'on');
hold(subplot3,'off');

%% FP Test
load([dir_test,'\Car_FP.mat'])
load([dir_test,'\Cyc_FP.mat'])
load([dir_test,'\Ped_FP.mat'])

ClS_Test_FP = [Car_FP_raw_sc;Cyc_FP_raw_sc;Ped_FP_raw_sc];

Y = ClS_Test_FP;
n = size(Y,1);
nc = length(label);
P = zeros(n,nc);
for k=1:n % object
    for clas=1:nc % classes
        for i=1:size(Valores,2)
            if (BinEdgesLow(clas,i) <= Y(k,clas)) & (Y(k,clas) < BinEdgesHigh(clas,i))
                P(k,clas) = Valores(clas,i);
            end
        end
    end
end
ML_FP = [];

ObjnessCar_FP = Car_FP_ObjSc;
ObjnessCyc_FP = Cyc_FP_ObjSc;
ObjnessPed_FP = Ped_FP_ObjSc;

ML_FP = P+lambda;
ML_FP = ML_FP./sum(ML_FP,2);
ML_FP(1:size(ObjnessCar_FP,1),1) = ML_FP(1:size(ObjnessCar_FP,1),1).*ObjnessCar_FP;
ML_FP(size(ObjnessCar_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1),2) = ML_FP(size(ObjnessCar_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1),2).*ObjnessCyc_FP;
ML_FP(size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+size(ObjnessPed_FP,1),3) = ML_FP(size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+size(ObjnessPed_FP,1),3).*ObjnessPed_FP;

figure3 = figure('Color',[1 1 1]);
subplot1 = subplot(1,3,1,'Parent',figure3);

hold(subplot1,'on');
histogram(ML_FP(1:size(ObjnessCar_FP,1),1),'Parent',subplot1,'Normalization','probability','NumBins',nbins);
title({'ML-FP-Car'});
box(subplot1,'on');
grid(subplot1,'on');
hold(subplot1,'off');

subplot2 = subplot(1,3,2,'Parent',figure3);
hold(subplot2,'on');
histogram(ML_FP(size(ObjnessCar_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1),2),'Parent',subplot2,'Normalization','probability','NumBins',nbins);
title({'ML-FP-Cyc'});
box(subplot2,'on');
grid(subplot2,'on');
hold(subplot2,'off');

% Create subplot
subplot3 = subplot(1,3,3,'Parent',figure3);
hold(subplot3,'on');
histogram(ML_FP(size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+1:size(ObjnessCar_FP,1)+size(ObjnessCyc_FP,1)+size(ObjnessPed_FP,1),3),'Parent',subplot3,'Normalization','probability','NumBins',nbins);
title({'ML-FP-Ped'});
box(subplot3,'on');
grid(subplot3,'on');
hold(subplot3,'off');
