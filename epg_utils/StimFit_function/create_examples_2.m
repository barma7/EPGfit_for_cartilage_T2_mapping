%% CREATE CPMG Examples
clear all; close all;
% IMPORT PULSE PROFILES
homeFldr = 'C:\ShareFldr-Ubuntu\Lab-work\Code\Projects\EPG-fitting-OAI-MESE';
exc = 90.*readmatrix(fullfile(homeFldr,'slice-profile-pulses','SIMENS\TBW4\SLR', 'excitation-slice-profile.txt'));
ref = 180.*readmatrix(fullfile(homeFldr,'slice-profile-pulses', 'SIMENS\TBW4\SLR', 'refocusing-slice-profile.txt'));


SvFldr_home = 'C:\ShareFldr-Ubuntu\Lab-work\2022\EPG-fitting-OAI-MESE/sim-data';
exp_fldr = 'dictionaries\SINC_TBW4\SLR\random_uniform\for_sim_exp\b1_uniform_10_110';

if ~exist(fullfile(SvFldr_home, exp_fldr), 'dir')
    mkdir(fullfile(SvFldr_home, exp_fldr))
end

%% Sequence params

EchoSpacing = 10; %in ms (milliseconds)
ETL = 7;
% Set options
opt.esp = EchoSpacing;
opt.etl = ETL;
opt.mode = 's';
opt.RFe.alpha = exc.*(pi/180);
opt.RFr.alpha = ref.*(pi/180);
opt.T1 = 1200;
opt.Nz = size(ref,2);
opt.debug = 0;

%% Uniform Sampling
Nb_exmpls = 4000;
lowT2 = 10;
highT2 = 110;
lowB1 = 0.6;
highB1 = 1.2;

%rng(5); % to assure reproducibility (optional)
T2s = lowT2 + (highT2 - lowT2).*rand(Nb_exmpls,1);
%T2s = CreateLutHalfGaussian(lowT2, highT2, Nb_exmpls);
B1s = lowB1 + (highB1 - lowB1).*rand(Nb_exmpls,1);

LUT = cat(2,T2s, B1s);

dictionary = zeros(Nb_exmpls,ETL);

T2s = LUT(:,1);
B1s = LUT(:,2);

tic
parfor i = 1:Nb_exmpls
    dictionary(i,:) = FSEsig(T2s(i),B1s(i),1,opt);
end
toc

writematrix(LUT, fullfile(SvFldr_home,exp_fldr,'LUT.txt'));
writematrix(ETL, fullfile(SvFldr_home,exp_fldr,'ETL.txt'));
writematrix(exc, fullfile(SvFldr_home,exp_fldr,'exc_profile.txt'));
writematrix(ref, fullfile(SvFldr_home,exp_fldr,'ref_profile.txt'));

save(fullfile(SvFldr_home,exp_fldr,'dictionary.mat'),'dictionary')