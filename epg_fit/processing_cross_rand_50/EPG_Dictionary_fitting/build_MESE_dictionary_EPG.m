function [dictionary, tElapsed, LUT] = build_MESE_dictionary_EPG(opt, T2s, B1s)

Nb_exmpls = length(T2s)*length(B1s);

LUT = zeros(Nb_exmpls,2);
count = 1;
for i=1:length(T2s)
    for k = 1: length(B1s)
        LUT(count,:) = cat(2,T2s(i), B1s(k));
        count = count+1;
    end
end

dictionary = zeros(Nb_exmpls,opt.etl);

T2s = LUT(:,1);
B1s = LUT(:,2);

tstart = tic;
for i = 1:Nb_exmpls
    dictionary(i,:) = FSEsig(T2s(i),B1s(i),1,opt);
end
tElapsed = toc(tstart); 
