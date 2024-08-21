function [c, Rsq, Nstd] = fit_linear_coefficient_EPG(p, ydata, opt)

T2 = p(1);
B1 = p(2);

signal = FSEsig(T2,B1,1,opt);

A = signal;

c = A\ydata;

% calculate R squared
yEst = A*c;

SStot = sum((ydata - mean(ydata)).^2);            % Total Sum-Of-Squares
%SSreg = sum((yEst - mean(ydata)).^2);             % Regression Sum-Of-Squares
SSres = sum((yEst - ydata).^2);                   % Residual Sum-Of-Squares
Rsq = 1-SSres/SStot;  

Nstd = std(ydata - yEst);