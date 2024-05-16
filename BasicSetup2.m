close all;
clear all;
clc;
% See available adapters for video

adV    = imaqhwinfo;
ad1    = adV.InstalledAdaptors{4};
wInfoV = imaqhwinfo(ad1);
numD   = size(wInfoV.DeviceIDs,2);
devIDU = 3;
devIDU = min(numD,devIDU);

vid1   = videoinput(ad1,devIDU);

% Init frame grabber
start(vid1);
wait(vid1,Inf);

% Retrieve the frames and timestamps for each frame.
[frames,time] = getdata(vid1, get(vid1,'FramesAvailable'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate frame rate by averaging difference between...
% each frame's timestamp
framerate = mean(1./diff(time));

 for idxF =1:size(frames,4)
     imshow(frames(:,:,:,idxF))
     pause(0.1)
 end
return
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recovery by image
% video1 = imaqhwinfo('winvideo',devIDU);
set(0,'Units','pixels') 
scSz = get(0,'ScreenSize');
%fP   = figure;
%fD   = figure;
%set(fD,'Visible', 'Off', 'Position',[0 0 1 1])   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Locate pattern on second
if  numel(get(0, 'MonitorPositions'))/4==1
    
    scP = get(0,'MonitorPositions'); 
    pos1    = get(0,'MonitorPositions'); 
    
elseif numel(get(0, 'MonitorPositions'))/4==2
    
    scP = get(0,'MonitorPositions'); 
    pos1    = get(fP,'Position');
    %pos1(1) = ;
    pos1 = [ scSz(3)  , scP(1,4)-scP(2,4),...
             scP(2,3) - scP(1,3),...
             scP(1,4) + abs(scP(1,4)-scP(2,4))];
    %pos1(3:4) = ;
end

for idx = 1:20
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
    dummyC     = binornd(1,0.5,pos1(4),pos1(3));
    P(:,:,1,idx) = dummyC;
    P(:,:,2,idx) = dummyC;
    P(:,:,3,idx) = dummyC;

    cF = [pwd filesep];
    cI = ['C:' filesep filesep 'IrfanView' filesep]; 
    %cd(cI);
    %imwrite(P(:,:,3,idx),'dummy.png');
    gSNP = imresize(getsnapshot(vid1),[pos1(4) pos1(3)]);
    imwrite(gSNP,'dummy.png');
    eval(['!' cI 'i_view32.exe ' 'dummy.png' ' /fs &']);
    eval(['!' cI 'i_view32.exe ' '/killmesoftly &']);
    %[MSG,SYS]=system(['&' cI 'i_view32.exe ' cF 'dummy.bmp' ' /fs & timeout 1 & ' cI 'i_view32.exe ' '/killmesoftly']);
    %SYS
    %eval(['!' cI 'i_view32.exe ' '/killmesoftly']);
    %cd(cF);

    %pause(5)
    
%     C(:,:,:,idx) = getsnapshot(vid1);
%     figure(fD)
%     subplot(1,3,1);
%     imshow(C(:,:,:,idx))
%     subplot(1,3,2);
%     imshow(rgb2gray(C(:,:,:,idx)));
%     subplot(1,3,3);
%     imshow(P(:,:,1,idx));
%     %drawnow
    %pause(0.1)
end