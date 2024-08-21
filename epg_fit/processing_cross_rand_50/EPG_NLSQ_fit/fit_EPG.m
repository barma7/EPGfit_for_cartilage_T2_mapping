function yEst = fit_EPG(p, ydata, opt)

T2 = p(1);
B1 = p(2);

signal = FSEsig(T2,B1,1,opt);

A = signal;

c = A\ydata;

yEst = A*c;

%figure(2)
%plot(yEst, '--', "LineWidth", 1.5); hold on;

