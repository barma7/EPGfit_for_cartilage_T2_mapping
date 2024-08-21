function residual = compute_residual_EPG(p, ydata, opt)

T2 = p(1);
B1 = p(2);
%M0 = p(2);

s0 = FSEsig(T2,B1,1,opt)';

%s0 = M0.*abs(s0./norm(s0));
s0 = abs(s0./norm(s0));

residual = ydata - s0;



