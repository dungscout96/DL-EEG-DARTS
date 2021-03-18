% clear
% Add EEGLAB path
addpath('/expanse/projects/nemar/eeglab');
eeglab; close;

%try, parpool(23); end

folderout = '/expanse/projects/nemar/child-mind-restingstate-dung-v2';
fileNamesClosed = dir(fullfile(folderout, '*_eyesclosed.set'));
subjs = readtable('test_subjs.csv');
subjs = subjs.Var1;
N = length(subjs)

X_test = [];
Y_test = [];
subj_test = {};

for iFile=1:N
    try
    EEGeyesc = pop_loadset('filepath', folderout, 'filename', [subjs{iFile} '_eyesclosed.set']);
    
    tmpdata = EEGeyesc.data;

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
    
    % append to XOri
    X_test = cat(3,X_test, tmpdata);
    Y_test = [Y_test, repelem(EEGeyesc.gender, size(tmpdata,3))];
    subj_test = [subj_test repelem(string(EEGeyesc.subjID), size(tmpdata,3))];
    catch ME
	disp('subj not found');
    end
end
% save
save('child_mind_x_test_extra.mat','X_test','-v7.3');
save('child_mind_y_test_extra.mat','Y_test','-v7.3');
save('test_subj_extra.mat','subj_test','-v7.3');
