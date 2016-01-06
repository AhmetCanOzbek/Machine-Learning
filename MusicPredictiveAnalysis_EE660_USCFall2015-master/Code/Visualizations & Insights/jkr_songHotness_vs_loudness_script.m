close all; clear all; clc;
workspace;

dataVar = importdata('jkr_songHotness_vs_loudness_data.csv',',',1);

fitpoly1=fit(dataVar.data(:,1),dataVar.data(:,2),'poly1')
fitpoly2=fit(dataVar.data(:,1),dataVar.data(:,2),'poly2')
fitpoly3=fit(dataVar.data(:,1),dataVar.data(:,2),'poly3')

scatter(dataVar.data(:,1),dataVar.data(:,2),'.','y')
hold on;
h1 = plot(fitpoly1,'r');
set(h1, 'LineWidth',2)
h2 = plot(fitpoly2,'g--');
set(h2, 'LineWidth',2)
h3 = plot(fitpoly3,'b:');
set(h3, 'LineWidth',2)
legend([h1 h2 h3],'poly1','poly2','poly3');
title('Jack Knife Regression : Song Hotness vs Loudness');
hold off;