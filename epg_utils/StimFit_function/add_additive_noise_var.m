function [y, SNR_g] = add_additive_noise_var(FID,N0)
%
% The function add_wgn add white gaussian noise with a desired SNR. It
% Works only for 1D NMR experiments .
% Author: Marco Barbieri
% Last Modifed: 7/04/2017
%-----------------------INPUT PARAMETERS----------------------------------
% sig...............is the vector containing the signal
% SNR_lin...........is the Signal-to-Noise ratio in lin scale
%-----------------------OUTPUT PARAMETERS----------------------------------
% y.................is the vector containing the Noised FID
% SNR_calc..........is the Signal-to-Noise ratio evaluated by the in-build
%                   MATLAB function (for double chek)
%--------------------------------------------------------------------------

L = length(FID(1,:));
Esig = sum(abs(FID).^2, 2)./L; %Power of the Signal

Nbexmpls = size(FID,1);

if isreal(FID)
    noise = sqrt(N0).*randn(Nbexmpls, L);
    Enoise = sum(abs(noise).^2, 2)./L;
    SNR_g = Esig./N0;
    y = FID + noise;
else
    n1 = sqrt(N0/2).*randn(Nbexmpls, L);
    n2 = sqrt(N0/2).*randn(Nbexmpls, L);

    %power of the noise
    noise = complex(n1, n2);
    Enoise = sum(abs(noise).^2, 2)./L;
    %SNR_g = Esig./Enoise;
    SNR_g = Esig./N0;
    
    y = complex(real(FID) + n1, imag(FID) + n2);
    %snr_cal = snr(y,complex(n1,n2));
end



