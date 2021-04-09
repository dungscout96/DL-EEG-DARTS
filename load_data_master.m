function load_data_master(winLength, numChan, isSpectral, isTopo)
% clear
% Add EEGLAB path
addpath('/expanse/projects/nemar/eeglab');
eeglab; close;

%try, parpool(23); end

folderout = '/expanse/projects/nemar/child-mind-restingstate-preprocessed';
fileNamesClosed = dir(fullfile(folderout, '*_eyesclosed.set'));
female = readtable('female.csv');
female = female.Var1;
male = readtable('male.csv');
male = male.Var1;
N = length(female);
% choose training, validation, and test from different subjects.
N_test_subjs = ceil(N * 0.125);
N_val_subjs = ceil(N * 0.3125);
N_train_subjs = length(fileNamesClosed) - N_test_subjs - N_val_subjs;

max_sample_per_subj = 85;
X_train = cell(1,N_train_subjs*2*max_sample_per_subj); % 2 is number of genders
Y_train = cell(1,N_train_subjs*2*max_sample_per_subj);
X_val = cell(1,N_val_subjs*2*max_sample_per_subj);
Y_val = cell(1,N_val_subjs*2*max_sample_per_subj);
X_test = cell(1,N_test_subjs*2*max_sample_per_subj);
Y_test = cell(1,N_test_subjs*2*max_sample_per_subj);
test_subjID = cell(1,N_test_subjs*2*max_sample_per_subj);
% subjID = {};

% dimension of number of sample in the data. If topo map, 4 (rgb x samples), otherwise 3
% (chan x times x samples)
if isTopo, sample_dim = 4; else, sample_dim = 3; end

parfor iFile=0:N-1
    for gender=0:1
        if gender == 1
            % female
            EEGeyesc = pop_loadset('filepath', folderout, 'filename', [female{iFile} '_eyesclosed.set']);
        else
            % male
            EEGeyesc = pop_loadset('filepath', folderout, 'filename', [male{iFile} '_eyesclosed.set']);
        end
        
        % sub-sample using window length
        EEGeyesc = eeg_regepochs( EEGeyesc2, 'recurrence', winLength, 'limits', [0 winLength]);
        tmpdata = EEGeyesc.data;

        % If numChan is 24, sub-select channel. Otherwise assuming it's 128
        % which is the original data
        if numChan == 24
            % sub-select channel
            channel_map = {'Fp1', 22; 'Fp2', 9; 'F7', 33;'F3',24;'Fz', 11;'F4',124;'F8', 122;'FC3', 29;'FCz', 6;'FC4', 111;'T3', 45;'C3', 36;
                'C4', 104;'T4', 108;'CP3', 42;'CPz', 55;'CP4', 93;'T5', 58;'P3', 52;'Pz', 62;'P4', 92;'T6', 96;'O1', 70; 'Cz', 'Cz'};
            chanindices = [];
            for iChannel = 1:size(channel_map,1)
                if ~ischar(channel_map{iChannel,2})
                    egiChannel = sprintf('E%d', channel_map{iChannel,2});
                    chanindices = [chanindices find(cellfun(@(x) strcmp(x,egiChannel), {EEGeyesc.chanlocs.labels}))];
                else
                    chanindices = [chanindices find(cellfun(@(x) strcmp(x,'Cz'), {EEGeyesc.chanlocs.labels}))];
                end
            end
            if (length(chanindices) < 24)
            warning('%s have missing channels. Skipped', fileNamesClosed(iFile).name);
                disp(size(tmpdata));
            continue;
            end 
            tmpdata = tmpdata(chanindices,:,:);
        end
        
        % If compute spectral
        if isSpectral
            freqRanges = [4 7; 7 13; 14 30]; % frequencies, but also indices
            % compute spectrum
            srates = 100;
            [XSpecTmp,~] = spectopo(tmpdata, winLength*1000, srates, 'plot', 'off', 'overlap', 50);
            XSpecTmp(:,1) = []; % remove frequency 0

            % get frequency bands
            theta = mean(XSpecTmp(:, freqRanges(1,1):freqRanges(1,2)), 2);
            alpha = mean(XSpecTmp(:, freqRanges(2,1):freqRanges(2,2)), 2);
            beta  = mean(XSpecTmp(:, freqRanges(3,1):freqRanges(3,2)), 2);
            
            %TODO: modify tmpdata to contains spectral and phase
            
            if isTopo
                % get grids
                [~, gridTheta] = topoplot( theta, EEGeyesc.chanlocs, 'verbose', 'off', 'gridscale', 28, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
                [~, gridAlpha] = topoplot( alpha, EEGeyesc.chanlocs, 'verbose', 'off', 'gridscale', 28, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
                [~, gridBeta ] = topoplot( beta , EEGeyesc.chanlocs, 'verbose', 'off', 'gridscale', 28, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
                gridTheta = gridTheta(end:-1:1,:); % for proper imaging using figure; imagesc(grid);
                gridAlpha = gridAlpha(end:-1:1,:); % for proper imaging using figure; imagesc(grid);
                gridBeta  = gridBeta( end:-1:1,:); % for proper imaging using figure; imagesc(grid);

                topoTmp = gridTheta;
                topoTmp(:,:,3) = gridBeta;
                topoTmp(:,:,2) = gridAlpha;
                tmpdata = single(topoTmp);
                
                % remove Nan
                minval = nanmin(nanmin(tmpdata,[],1),[],2);
                maxval = nanmax(nanmax(tmpdata,[],1),[],2);
                
                % transform to RGB image
                tmpdata = bsxfun(@rdivide, bsxfun(@minus, tmpdata, minval), maxval-minval)*255;
                tmpdata(isnan(tmpdata(:))) = 0;
            end
        end
        
        % append to XOri
        for sample=1:size(tmpdata,sample_dim)
            idx = iFile*2*max_sample_per_subj+gender*max_sample_per_subj+sample;
            if (iFile <= N_test_subjs)
                X_test{idx} = tmpdata(:,:,sample);
                Y_test{idx} = EEGeyesc.gender;
                test_subjID{idx} = string(EEGeyesc.subjID);
            elseif (iFile <= N_test_subjs + N_val_subjs)
                X_val{idx} = tmpdata(:,:,sample);
                Y_val{idx} = EEGeyesc.gender;
            else
                X_train{idx} = tmpdata(:,:,sample);
                Y_train{idx} = EEGeyesc.gender;
            end
        end
    end
end

X_test = cat(sample_dim,X_test{:});
Y_test = cat(1,Y_test{:});
test_subjID = cat(1, test_subjID{:});
X_val = cat(sample_dim,X_val{:});
Y_val = cat(1,Y_val{:});
X_train = cat(sample_dim,X_train{:});
Y_train = cat(1,Y_train{:});

% save
param_text = ['_' num2str(winLength) 's'];
param_text = [param_text '_' num2str(numChan) 'chan'];
if isSpectral
    if isTopo
        param_text = [param_text '_topo'];
    else
        param_text = [param_text '_spectral'];
    end
else
    param_text = [param_text '_raw'];
end
save(['child_mind_x_train' param_text '.mat'],'X_train','-v7.3');
save(['child_mind_y_train' param_text '.mat'],'Y_train','-v7.3');
save(['child_mind_x_val' param_text '.mat'],'X_val','-v7.3');
save(['child_mind_y_val' param_text '.mat]'],'Y_val','-v7.3');
save(['child_mind_x_test' param_text '.mat'],'X_test','-v7.3');
save(['child_mind_y_test' param_text '.mat'],'Y_test','-v7.3');
save('test_subj.mat','test_subjID','-v7.3');

