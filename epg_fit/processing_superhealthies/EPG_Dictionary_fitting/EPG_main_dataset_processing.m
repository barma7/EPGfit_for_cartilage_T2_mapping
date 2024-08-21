clear all; close all;

%% DEFINE SOME PATHS
% Add EPG simulation to MATLAB path
home_uitls_path = "/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/code/epg_utils";
epg_path = fullfile(home_uitls_path,"StimFit_function");
addpath(epg_path);

% pulse path
pulse_path = fullfile(home_uitls_path,"Pulses_and_SliceProfiles/SINC_pulses/TWB2"););

home_path_dictionary = fullfile(home_uitls_path, "sim-data/dictionaries");
sv_name_dict = "SINC/TBW2/SLR/grid";

% saving path for storing the processed maps
sv_name = "epg_dictionary";

% home data and save folders
home_path = '/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/superhealthies';
home_save_path = '/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/superhealthies';

%% LOAD SUBJECT FOLDERS
a = dir(fullfile(home_path, '9*'));

nifti_name = "t2_4d_array.nii";
mask_name = "registered_dess_segmentation.nii";

t2map_name = "t2.nii";
b1map_name = "b1.nii";
r2map_name = "r2.nii";
snrmap_name = "nSNR.nii";

%% DEFINE AND LOAD DICTIONARIES
dict_folder = fullfile(home_path_dictionary, sv_name_dict);
if ~exist(fullfile(dict_folder, 'dictionary.mat'), 'file')
    
    Dinfo = readtable(fullfile(pulse_path, "PulseSpecs.csv"), 'VariableNamingRule','preserve');
    exc = readmatrix(fullfile(pulse_path,"90", "SLR", 'pulse_profile.txt'),'delimiter',' ')';
    ref = readmatrix(fullfile(pulse_path,"180", "SLR", 'pulse_profile.txt'),'delimiter',' ')';
  
    ETL = 7;
    EchoSpacing = 10; % [ms]
    
    % Set options for pulse sequence param and cartilage tissue 
    opt.esp = EchoSpacing;
    opt.etl = ETL;
    opt.mode = 's';
    opt.RFe.alpha = exc;
    opt.RFr.alpha = ref;
    opt.T1 = 1200;
    opt.Nz = size(ref,2);
    opt.debug = 0;

    disp("Building dictionary using EPG model...")
    % CREATE DICTIONARY FOR FAT CALIBRATION 
    T2s = 10:0.25:100;
    B1s = 0.4:0.02:1.2;
    [dictionary, tElapsed_dictionary, LUT] = build_MESE_dictionary_EPG(opt, T2s, B1s);
    disp(strcat("Dictionary created in: ", strcat(num2str(tElapsed_dictionary), "s")));

    % Save Dictionary w/ info and LUT
    if ~exist(fullfile(dict_folder), 'dir')
        mkdir(fullfile(dict_folder))
    end
    save(fullfile(dict_folder,'dictionary.mat'),'dictionary');
    writetable(Dinfo, fullfile(dict_folder,'PulseInfo.csv'));
    writematrix(LUT, fullfile(dict_folder,'LUT.txt'));
else
    load(fullfile(dict_folder, 'dictionary.mat'));
    LUT = readmatrix(fullfile(dict_folder, 'LUT.txt'));
end 

dim = size(dictionary);

% Normalize Dictionary
Dict_nor = single(zeros(size(dictionary)));
for i=1:length(Dict_nor(:,1))
    Dict_nor(i,:) = dictionary(i,:)./norm(dictionary(i,:));
end

strt = tic;
idx = 1;
for k=1:length(a)
    sub = a(k).name;
    disp(cat(2,'Processing subject: ', sub))
    
    %list subfolders
    subfldrs = dir(fullfile(a(k).folder, a(k).name));
    subfldrs(1:2) = [];
    
    % for each subfodler
    for sk=1:length(subfldrs)
        % define folder
        time_id = subfldrs(sk).name;
        disp(time_id);
        dataset_folder = fullfile(home_path, sub, time_id);
        mask_folder = fullfile(home_path, sub, time_id);
        
        % process data only if data exist in folder
        if isfile(fullfile(dataset_folder, nifti_name))
            
            svFldr = fullfile(home_save_path, sub, time_id, sv_name);
            
            if ~exist(fullfile(svFldr), 'dir')
                mkdir(fullfile(svFldr))
            end
            
            % LOAD DATA
            data_nifti = squeeze(single(niftiread(fullfile(dataset_folder,nifti_name))));
            region_mask = single(niftiread(fullfile(mask_folder,mask_name)));
            info_nifti = niftiinfo(fullfile(dataset_folder,nifti_name));

            % Check if segmentation has right number of labels
            seg_orig_set = unique(region_mask(:));
            seg_target_set = [1,2,3];
            % process data only if all 3 labels are present 
            if sum(ismember(seg_orig_set, seg_target_set)) == 3
            
                % DEFINE SOME SPECS
                nb_row = size(data_nifti,1);
                nb_col = size(data_nifti,2);
                nb_slices = size(data_nifti,3);
                ETL = size(data_nifti,4);
                
                T2map = single(zeros(nb_row*nb_col*nb_slices,1));
                B1map = single(zeros(nb_row*nb_col*nb_slices,1));
                R2map = single(zeros(nb_row*nb_col*nb_slices,1));
                SNRmap = single(zeros(nb_row*nb_col*nb_slices,1));
                
                % FLAT DATA AND MASK 
                data = reshape(data_nifti, nb_row*nb_col*nb_slices, ETL);
                mask_flat = region_mask(:);
                
                non_zero_indexes = find((mask_flat > 0) & (mask_flat < 4));
                
                data_sampled = data(non_zero_indexes,:);
                
                % Normalize data
                data_n = single(zeros(size(data_sampled)));
                for i=1:length(data_sampled(:,1))
                    data_n(i,:) = data_sampled(i,:)./norm(data_sampled(i,:));
                end
                
                [T2fit, B1fit, data_est, tElapsed] = dictionary_matching_dot(LUT, Dict_nor, data_n);
                disp(['Fitted non zero element in volume in : ', num2str(tElapsed)]);
                list_time(idx) = tElapsed;
                
                % Evaluate Rsquare (goodness of fit)
                SStot = sum((data_n - mean(data_n,2)).^2, 2);            % Total Sum-Of-Squares
                SSres = sum((data_n - data_est).^2, 2);                   % Residual Sum-Of-Squares
                Rsq = 1-SSres./SStot;  
                SNR = (sum(abs(data_n).^2, 2)./size(data_n,2))./std(data_est - data_n,[],2);

                T2map(non_zero_indexes,1) = T2fit;
                B1map(non_zero_indexes,1) = B1fit;
                R2map(non_zero_indexes) = Rsq;
                SNRmap(non_zero_indexes) = SNR;

                T2map = reshape(T2map, nb_row, nb_col, nb_slices);
                B1map = reshape(B1map, nb_row, nb_col, nb_slices);
                R2map = reshape(R2map, nb_row, nb_col, nb_slices);
                SNRmap = reshape(SNRmap, nb_row, nb_col, nb_slices);

                %Save Fitted Values
                writematrix(T2fit, fullfile(svFldr, 'T2fit.txt'));
                writematrix(B1fit, fullfile(svFldr, 'B1fit.txt'));
                writematrix(Rsq, fullfile(svFldr, 'Rsqfit.txt'));
                writematrix(SNR, fullfile(svFldr, 'snrs.txt'));
                        
                % save maps
                info_out = info_nifti;
                info_out.ImageSize = info_out.ImageSize(1:3);
                info_out.PixelDimensions = info_out.PixelDimensions(1:3);
                info_out.Datatype = 'single';
                
                niftiwrite(T2map, fullfile(svFldr, t2map_name), info_out);
                niftiwrite(B1map, fullfile(svFldr, b1map_name), info_out);
                niftiwrite(single(R2map), fullfile(svFldr, r2map_name),info_out);
                niftiwrite(single(SNRmap), fullfile(svFldr, snrmap_name),info_out);

                msg = "Subject processed successfully";
                status = 2;
            else
                msg = "Subject does not have all the required labels, skipping subject";
                list_time(idx) = 0;
                disp(msg)
                status = 1;
            end
        else
            msg = "File does not exist, skipping subject";
            list_time(idx) = 0;
            disp(msg)
            status = 0;
        end

        list_sub_id(idx) = str2double(sub);
        list_status(idx) = status;
        list_msg(idx) = msg;
        list_time_id(idx) = str2double(time_id);
        idx = idx + 1;
    end
end
toc(strt);
varNames = ["sub_id", "time_id", "status", "processing time", "message"];

table = table(list_sub_id', list_time_id', list_status', list_time' ...
    , list_msg');
table.Properties.VariableNames = varNames;

writetable(table, fullfile(home_save_path, 'processing_log_EPG_dictionary.csv'));

rmpath(epg_path);
        

        
        
        




