%% Gledson Melotti
clear;
close all
clc

dir_probability = '...\KITTI\RGB\';
load([dir_probability,'Probability_logits_Train.mat']);
load([dir_probability,'Train_labels.mat']);
load([dir_probability,'Probability_logits_test.mat']);
load([dir_probability,'Test_labels.mat']);
load([dir_probability,'Test_predict.mat']);
load([dir_probability,'Probability.mat']);
Probability_test=Probability;

c = unique(Train_labels);
nc = length(c); %3
ctr1 = Train_labels == c(1); ctr2 = Train_labels == c(2); ctr3 = Train_labels == c(3);
nbins=25;

%% Normalized Histogram 
figure(1); cla; hold on
hc1 = histogram(Probability_logits_Train(ctr1,1),nbins,'Normalization','probability','FaceColor','b'); %class=c1 / y=c1
Valores1 = hc1.Values;
BinEdgesLow1 = hc1.BinEdges(1:nbins);
BinEdgesHigh1 = hc1.BinEdges(2:nbins+1);

hc2 = histogram(Probability_logits_Train(ctr2,2),nbins,'Normalization','probability','FaceColor','r'); %class=c1 / y=c2
Valores2 = hc2.Values;
BinEdgesLow2 = hc2.BinEdges(1:nbins);
BinEdgesHigh2 = hc2.BinEdges(2:nbins+1);

hc3 = histogram(Probability_logits_Train(ctr3,3),nbins,'Normalization','probability','FaceColor','g'); %class=c1 / y=c3
Valores3 = hc3.Values;
BinEdgesLow3 = hc3.BinEdges(1:nbins);
BinEdgesHigh3 = hc3.BinEdges(2:nbins+1);
grid

%% Norma distribution: mean and variance
pdf1 = fitdist(Probability_logits_Train(ctr1,1),'Normal');
pdf2 = fitdist(Probability_logits_Train(ctr2,2),'Normal');
pdf3 = fitdist(Probability_logits_Train(ctr3,3),'Normal');

ypdf1 = pdf(pdf1,Probability_logits_Train(ctr1,1));
[~, idx1] = sort(Probability_logits_Train(ctr1,1));
x1 = Probability_logits_Train(ctr1,1);

ypdf2 = pdf(pdf2,Probability_logits_Train(ctr2,2));
[~, idx2] = sort(Probability_logits_Train(ctr2,2));
x2 = Probability_logits_Train(ctr2,2);

ypdf3 = pdf(pdf3,Probability_logits_Train(ctr3,3));
[~, idx3] = sort(Probability_logits_Train(ctr3,3));
x3 = Probability_logits_Train(ctr3,3);

figure(2)
plot(x1(idx1),ypdf1(idx1),'b','LineWidth',2);
hold on
plot(x2(idx2),ypdf2(idx2),'r','LineWidth',2);
hold on
plot(x3(idx3),ypdf3(idx3),'g','LineWidth',2);
grid
legend('Pedestrian','Car','Cyclist')

%% ML
Y = Probability_logits_test;
n = size(Y,1); nc = 3;
ML = zeros(n,nc);

P1 = zeros(n,1);
P2 = zeros(n,1);
P3 = zeros(n,1);

for k=1:n
    for i=1:size(Valores1,2)
        if (BinEdgesLow1(i) <= Y(k,1)) & (Y(k,1) < BinEdgesHigh1(i))
           P1(k) = Valores1(i);
        end
    end
    for i=1:size(Valores2,2)
        if (BinEdgesLow2(i) <= Y(k,2)) & (Y(k,2) < BinEdgesHigh2(i))
           P2(k) = Valores2(i);
        end
    end
    for i=1:size(Valores3,2)
        if (BinEdgesLow3(i) <= Y(k,3)) & (Y(k,3) < BinEdgesHigh3(i))
           P3(k) = Valores3(i);
        end
    end
end

%% ML
lambda = 0.01; % Additive Smoothing
P1_ML = lambda + P1;
P2_ML = lambda + P2;
P3_ML = lambda + P3;
N = P1_ML+P2_ML+P3_ML;
ML = [P1_ML,P2_ML,P3_ML]./N;

%% MAP
MAP = zeros(n,nc);
% Gaussian by class
% pdf1, pdf2 e pdf3 are training data and Y is the test probability
prior1 = pdf(pdf1,Y(:,1));
prior2 = pdf(pdf2,Y(:,2)); 
prior3 = pdf(pdf3,Y(:,3));

lambda = 0.01; % Additive Smoothing

P1_MAP = lambda+(double(prior1).*P1); 
P2_MAP = lambda+(double(prior2).*P2); 
P3_MAP = lambda+(double(prior3).*P3);
N = P1_MAP + P2_MAP + P3_MAP;
MAP = [P1_MAP, P2_MAP, P3_MAP]./N;

%% Performance measures / results
[LTe, iTest]=unique(Test_labels);
cte1 = Test_labels == LTe(1); cte2 = Test_labels == LTe(2); cte3 = Test_labels == LTe(3);
NC1 = sum(cte1); NC2 = sum(cte2); NC3 = sum(cte3);
Y = Probability_test;

%% Base line: SoftMax
TP = zeros(1,3);
FP = zeros(1,3);
FN = zeros(1,3);
yFP1 = [0, 0];
i1 = 1; i2 = [2 3];
for i=1:(iTest(2)-1)
    res = Y(i,:) == max(Y(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP1(1) =  yFP1(1) + Y(i,res);
       yFP1(2) =  yFP1(2) + 1;
    end    
end
FN(i1) = NC1 - TP(i1);

i1 = 2; i2 = [1 3];
for i=iTest(2):(iTest(3)-1)
    res = Y(i,:) == max(Y(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP1(1) =  yFP1(1) + Y(i,res);
       yFP1(2) =  yFP1(2) + 1;
    end    
end
FN(i1) = NC2 - TP(i1);

i1 = 3; i2 = [1 2];
for i=iTest(3):n
    res = Y(i,:) == max(Y(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP1(1) =  yFP1(1) + Y(i,res);
       yFP1(2) =  yFP1(2) + 1;
    end    
end
FN(i1) = NC3 - TP(i1);
disp('********** Baseline ');
Pre = TP./(TP + FP);
Rec = TP./(TP + FN);
F1 = 2*(Pre.*Rec)./(Pre + Rec);
fprintf('F-1 = %1.2f \n',100*mean(F1));
TP1 = TP; FP1 = FP; FN1 = FN;

%% ML
TP = zeros(1,3);
FP = zeros(1,3);
FN = zeros(1,3);
yFP2 = [0, 0];
i1 = 1; i2 = [2 3];
for i=1:(iTest(2)-1)
    res = ML(i,:) == max(ML(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP2(1) =  yFP2(1) + ML(i,res);
       yFP2(2) =  yFP2(2) + 1;
    end
end
FN(i1) = NC1 - TP(i1);

i1 = 2; i2 = [1 3];
for i=iTest(2):(iTest(3)-1)
    res = ML(i,:) == max(ML(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP2(1) =  yFP2(1) + ML(i,res);
       yFP2(2) =  yFP2(2) + 1;
    end
end
FN(i1) = NC2 - TP(i1);

i1 = 3; i2 = [1 2];
for i=iTest(3):n
    res = ML(i,:) == max(ML(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP2(1) =  yFP2(1) + ML(i,res);
       yFP2(2) =  yFP2(2) + 1;
    end
end
FN(i1) = NC3 - TP(i1);

disp('********** ML ');
Pre = TP./(TP + FP);
Rec = TP./(TP + FN);
F1 = 2*(Pre.*Rec)./(Pre + Rec);
fprintf('F-1 = %1.2f \n',100*mean(F1));
TP2 = TP; FP2 = FP; FN2 = FN;

%% MAP
TP = zeros(1,3);
FP = zeros(1,3);
FN = zeros(1,3);
yFP3 = [0, 0];
i1 = 1; i2 = [2 3];
for i=1:(iTest(2)-1)
    res = MAP(i,:) == max(MAP(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP3(1) =  yFP3(1) + MAP(i,res);
       yFP3(2) =  yFP3(2) + 1;
    end
end
FN(i1) = NC1 - TP(i1);

i1 = 2; i2 = [1 3];
for i=iTest(2):(iTest(3)-1)
    res = MAP(i,:) == max(MAP(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP3(1) =  yFP3(1) + MAP(i,res);
       yFP3(2) =  yFP3(2) + 1;
    end
end
FN(i1) = NC2 - TP(i1);

i1 = 3; i2 = [1 2];
for i=iTest(3):n
    res = MAP(i,:) == max(MAP(i,:)); %1x3 logical array
    TP(i1) = TP(i1) + res(i1);
    FP(i2) = FP(i2) + res(i2);
    if sum(res(i2))==1
       yFP3(1) =  yFP3(1) + MAP(i,res);
       yFP3(2) =  yFP3(2) + 1;
    end
end
FN(i1) = NC3 - TP(i1);

disp('********** MAP ');
Pre = TP./(TP + FP);
Rec = TP./(TP + FN);
F1 = 2*(Pre.*Rec)./(Pre + Rec);
fprintf('F-1 = %1.2f \n',100*mean(F1));
TP3 = TP; FP3 = FP; FN3 = FN;

%% False Positive Rate
Neg = [NC2+NC3, NC1+NC3, NC1+NC2];
FPR1 = FP1./Neg;      %FP/Neg
FPR2 = FP2./Neg;      %FP/Neg
FPR3 = FP3./Neg;      %FP/Neg
disp('-------------------------');
fprintf('ave.FPR-Base = %1.2f \n',100*mean(FPR1));
fprintf('ave.FPR-ML = %1.2f \n',100*mean(FPR2));
fprintf('ave.FPR-MAP = %1.2f \n',100*mean(FPR3));

%%
figure(3);
set(gcf,'color','w');
data1 = MAP(cte1,2:3);
data2 = MAP(cte1,1); %positives

axes1 = axes('Tag','suptitle','Position',[0 1 1 1]);
axis off

text('Parent',axes1,'HorizontalAlignment','center','FontSize',14,...
    'Position',[0.5 -0.0294145105819197 0],...
    'Visible','on');

axes2 = axes('Position',[0.13, 0.11, 0.213405797101449, 0.815]);
hold(axes2,'on');

histogram(data1,'Parent',axes2,'Normalization','probability',...
    'NumBins',nbins);

histogram(data2,'Parent',axes2,'Normalization','probability',...
    'NumBins',nbins);
legend('Negatives','Positives')

data3 = MAP(cte2,[1,3]);
data4 = MAP(cte2,2);
grid(axes2,'on');
hold(axes2,'off');
% Create axes
axes3 = axes('Position',[0.38 0.11 0.213405797101449 0.815]);
hold(axes3,'on');

histogram(data3,'Parent',axes3,'Normalization','probability',...
    'NumBins',nbins);

% Create histogram
histogram(data4,'Parent',axes3,'Normalization','probability',...
    'NumBins',nbins);

legend('Negatives','Positives')
data5 = MAP(cte3,1:2);
data6 = MAP(cte3,3);
grid(axes3,'on');
hold(axes3,'off');
% Create axes
axes4 = axes('Position',[0.63 0.11 0.213405797101449 0.815]);
hold(axes4,'on');

% Create histogram
histogram(data5,'Parent',axes4,'Normalization','probability',...
    'NumBins',nbins);

histogram(data6,'Parent',axes4,'Normalization','probability',...
    'NumBins',nbins);
legend('Negatives','Positives')

grid(axes4,'on');
hold(axes4,'off');

%% Subfigure Normalized Histogram and PDF
figure(4); set(gcf,'color','w'); cla; hold on
% % % --- proper pdf
figure(4); set(gcf,'color','w'); cla;
subplot(1,2,1)
hc1 = histogram(Probability_logits_Train(ctr1,1),nbins,'Normalization','probability','facecolor','b'); %class=c1 / y=c1
hold on
hc2 = histogram(Probability_logits_Train(ctr2,2),nbins,'Normalization','probability','facecolor','r'); %class=c1 / y=c2
hold on
hc3 = histogram(Probability_logits_Train(ctr3,3),nbins,'Normalization','probability','facecolor','g'); %class=c1 / y=c3
legend('Pedestrian','Car','Cyclist')
grid
subplot(1,2,2)
plot(x1(idx1),ypdf1(idx1),'b','LineWidth',2);
hold on
plot(x2(idx2),ypdf2(idx2),'r','LineWidth',2);
hold on
plot(x3(idx3),ypdf3(idx3),'g','LineWidth',2);
grid
legend('Pedestrian','Car','Cyclist')
