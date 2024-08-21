%% Script to perform 3-component fitting of Cortical Bone, BoneMarrow and Trabecular bone
clear all; close all;

%% DEFINE SOME PATHS
% Add EPG simulation to MATLAB path
home_uitls_path = "/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/code/epg_utils";
epg_path = fullfile(home_uitls_path,"StimFit_function");
addpath(epg_path);

% pulse path
pulse_path = fullfile(home_uitls_path,"Pulses_and_SliceProfiles/SINC_pulses/TWB2");

%% PATHS TO DATA AND SAVE FOLDERS
% saving path for storing the processed maps
sv_name = "epg_nlsq";

home_path = '/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/cross_sectional_rand_50';
home_save_path = '/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/cross_sectional_rand_50';

a = dir(fullfile(home_path, '9*'));

nifti_name = "t2_4d_array.nii";
mask_name = "registered_dess_segmentation.nii";


%% PATHS FOR EPG SIMULATION FORMATION
Dinfo = readtable(fullfile(pulse_path, "PulseSpecs.csv"), 'VariableNamingRule','preserve');
exc = readmatrix(fullfile(pulse_path,"90", "SLR", 'pulse_profile.txt'),'delimiter',' ')';
ref = readmatrix(fullfile(pulse_path,"180", "SLR", 'pulse_profile.txt'),'delimiter',' ')';

%% Set some fit options
%   Numeric fitting options
opt.lsq.fopt = optimset('lsqcurvefit');
opt.lsq.fopt.TolX = 1e-3;     %   Fitting accuracy: 0.1 ms
opt.lsq.fopt.TolFun = 1.0e-9;
opt.lsq.fopt.MaxIter = 400;
opt.lsq.fopt.Display = 'off';

opt.lsq.Icomp.X0   = [0.035 0.99];      %   Starting point (1 x 2) [T2(s) B1(fractional)]
opt.lsq.Icomp.XU   = [0.110 1.20];      %   Upper bound (1 x 2)
opt.lsq.Icomp.XL   = [0.010 0.60];      %   Lower bound (1 x 2)

%parpool(12);

list_sub_id = zeros(length(a),1);
list_time = zeros(length(a),1);
list_status = zeros(length(a),1);
list_msg = zeros(length(a),1);

strt = tic;
for k=1:length(a)

    sub = a(k).name;
    disp(sub);

    dataset_folder = fullfile(home_path, sub);
    mask_folder = fullfile(home_path, sub);
    
    if isfile(fullfile(dataset_folder,nifti_name))

        svFldr = fullfile(home_save_path, sub, sv_name);
        
        if ~exist(fullfile(svFldr), 'dir')
            mkdir(fullfile(svFldr))
        end
        
        % LOAD DATA
        data_nifti = double(squeeze(single(niftiread(fullfile(dataset_folder,nifti_name)))));
        region_mask = double(niftiread(fullfile(mask_folder,mask_name)));
        info_nifti = niftiinfo(fullfile(dataset_folder,nifti_name));
   
        seg_list = unique(region_mask);
        target_list = [1, 2, 3];
        
        % Check if the segmentation has all target labels
        if all(ismember(target_list, seg_list))
        
            % DEFINE SOME SPECS
            nb_row = size(data_nifti,1);
            nb_col = size(data_nifti,2);
            nb_slices = size(data_nifti,3);
            ETL = size(data_nifti,4);
            EchoSpacing = 10e-3;
        
            % Set Parameter options
            opt.esp = EchoSpacing;
            opt.etl = ETL;
            opt.mode = 's';
            opt.RFe.alpha = exc;
            opt.RFr.alpha = ref;
            opt.T1 = 1.2;
            opt.Nz = size(ref,2);
            opt.debug = 0;
        
            % FLAT DATA AND MASK 
            data = reshape(data_nifti, nb_row*nb_col*nb_slices, ETL);
            mask_flat = region_mask(:);
            
            non_zero_indexes = find((mask_flat > 0) & (mask_flat < 4));
            
            data = data(non_zero_indexes,:);
        
            % clean non zero data
            [rowIdx,colIdx] = find(data == 0);
        
            if ~isempty(rowIdx)
                for i=1:length(rowIdx)
                    if (colIdx(i) < 7) && (colIdx(i) > 1)
                        data(rowIdx(i), colIdx(i)) = mean([data(rowIdx(i), colIdx(i)-1), data(rowIdx(i), colIdx(i)+1)]);
                    elseif colIdx(i) == 1
                        data(rowIdx(i), colIdx(i)) = data(rowIdx(i), colIdx(i)+1);
                    else
                        data(rowIdx(i), colIdx(i)) = data(rowIdx(i), colIdx(i)-1);
                    end
                end
            end
        
            % Clean NaN values
            [rowIdxN,colIdxN] = find((isnan(data) == 1) | (data == 0));
            if ~isempty(rowIdxN)
                for i=1:length(rowIdxN)
                    data(rowIdxN(i),:) = zeros(1,ETL);
                end
            end
        
            %% Perform fitting voxel-wise
            % x = (M0, T2, B1)
            Nb_examples = size(data,1);
        
            x_est = zeros(Nb_examples,2);
            rhos = zeros(Nb_examples,1);
            Rsq = zeros(Nb_examples,1);
        
            ub = opt.lsq.Icomp.XU;
            lb = opt.lsq.Icomp.XL;
            
            x0 = opt.lsq.Icomp.X0;

            tstart = tic;
            parfor i=1:Nb_examples
                ydata = data(i,:)';
        
                %figure(2)
                %plot(ydata,"LineWidth",2,"Color","black"); hold on;
        
                % Inizialize starting point
                F = @(x, varargin)fit_EPG(x, ydata, opt);
        
                x_est(i,:)= lsqcurvefit(F,x0, opt, ydata, lb, ub, opt.lsq.fopt);
                
                %Estimate S0
                [rhos(i), Rsq(i)] = fit_linear_coefficient_EPG(x_est(i,:), ydata, opt);
                %close(2);
            end
            tElapsed = toc(tstart);
           
            %% Reproduce maps and save images 
            
            m0 = abs(rhos);   
            T2s = x_est(:,1).*1000; %[ms]
            T2s( T2s<= 0) = NaN;
            m0( T2s <= 0) = NaN;
            Rsq( T2s <= 0) = NaN;
        
            T2 = zeros(nb_row*nb_col*nb_slices,1);
            T2(non_zero_indexes) = T2s;
            
            M0 = zeros(nb_row*nb_col*nb_slices,1);
            M0(non_zero_indexes) = m0;
            
            B1 = zeros(nb_row*nb_col*nb_slices,1);
            B1(non_zero_indexes) = x_est(:,2);
        
            R2 = zeros(nb_row*nb_col*nb_slices,1);
            R2(non_zero_indexes) = Rsq;
        
            %Save Fitted Values
            writematrix(T2s, fullfile(svFldr, 'T2fit.txt'));
            writematrix(Rsq, fullfile(svFldr, 'Rsqfit.txt'));
            
            info = info_nifti;
            info.Datatype = 'double';
            info.PixelDimensions = info.PixelDimensions(1:3);
            info.ImageSize = info.ImageSize(1:3);
            % recreate 3D arrays
            niftiwrite(reshape(T2, nb_row, nb_col, nb_slices), fullfile(svFldr,'t2.nii'), info);
            niftiwrite(reshape(B1, nb_row, nb_col, nb_slices), fullfile(svFldr,'b1.nii'), info);
            niftiwrite(reshape(M0, nb_row, nb_col, nb_slices), fullfile(svFldr,'m0.nii'), info);
            niftiwrite(reshape(R2, nb_row, nb_col, nb_slices), fullfile(svFldr,'r2'), info);
            msg = "Subject processed successfully";
            status = 2;
            list_time(k) = tElapsed;
        else
            msg = "Subject does not have all the required labels, skipping subject";
            status = 1;
            list_time(k) = 0;
        end
    else
        msg = "file does not exist...skipping";
        status = 0;
        list_time(k) = 0;
    end
    disp(msg)
    list_sub_id(k) = str2double(sub);
    list_status(k) = status;
    list_msg(k) = msg;
end
toc(strt);

colnames = {'sub_id', 'status', 'processing time', 'message'};
T = table(list_sub_id, list_status, list_time, list_msg, 'VariableNames',colnames);

writetable(T, fullfile(home_save_path, 'processing_log_EPG_NLSQ.csv'));

exit;

