%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Santibáñez Peet 2014
%  fsantibanezleal@ug.uchile.cl

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Take current path
currentPath = [pwd filesep];
folderCode  = currentPath;
cF = [pwd filesep];
cI = ['C:' filesep filesep 'IrfanView' filesep]; 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Provide default params
% See available adapters for video
adV    = imaqhwinfo;
ad1    = adV.InstalledAdaptors{4};
wInfoV = imaqhwinfo(ad1);
numD   = size(wInfoV.DeviceIDs,2);
devIDU = 3;
devIDU = min(numD,devIDU);
vid1   = videoinput(ad1,devIDU);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Specific params for speceific computers
% provide PC name in order to use specific paths...
[~, pcName] = system('hostname');
pcName      = char(double(pcName(1:end-1)));
Friend      = computer;
    if strcmp(Friend,'MAC2')
        disp(['Please' '... try to use a real operative system'])
    elseif isunix

    elseif strcmp(Friend(1:2),'PC')
        if strcmp(pcName,'CS_Server')
            folderDB   = [''];
            folderCode = [''];
        elseif  strcmp(pcName,'CS_SOFI')
            % See available adapters for video
            adV    = imaqhwinfo;
            ad1    = adV.InstalledAdaptors{1};
            wInfoV = imaqhwinfo(ad1);
            numD   = size(wInfoV.DeviceIDs,2);
            % I have 3 available webcams
            % 1: Frontal Camera
            % 2: Back Camera
            % 3: USB webcam
            devIDU = 3;
            devIDU = min(numD,devIDU);
            vid1   = videoinput(ad1,devIDU);
        else
            % Peet PC
            adV    = imaqhwinfo;
            ad1    = adV.InstalledAdaptors{4};
            wInfoV = imaqhwinfo(ad1);
            numD   = size(wInfoV.DeviceIDs,2);
            devIDU = 3;
            devIDU = min(numD,devIDU);
            vid1   = videoinput(ad1,devIDU);

            %disp('Please provide right paths!!!!!; ')
            %return
        end
    else
        disp('I don*t recognize this computer; ')
        disp('aborted mision... please contact God.')
        disp('Ragnarok started!!!!!!!!!! RUN!!!!!.')
    end

addpath(folderCode);
addpath([folderCode 'Functions']);


