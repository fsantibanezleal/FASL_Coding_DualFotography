close all;
clear;
% See available adapters for video

adV    = imaqhwinfo;
ad1    = adV.InstalledAdaptors{4};
wInfoV = imaqhwinfo(ad1);
numD   = size(wInfoV.DeviceIDs,2);
devIDU = 2;
devIDU = min(numD,devIDU);

vid1   = videoinput(ad1,devIDU)

% Init frame grabber
start(vid1);
wait(vid1,Inf);

% Retrieve the frames and timestamps for each frame.
[frames,time] = getdata(vid1, get(vid1,'FramesAvailable'));

% Calculate frame rate by averaging difference between...
%each frame's timestamp
framerate = mean(1./diff(time))

% for idxF =1:size(frames,4)
%     imshow(frames(:,:,:,idxF))
%     pause(0.1)
% end

%% recovery by image
%video1 = imaqhwinfo('winvideo',devIDU);
set(0,'Units','pixels') 
scSz = get(0,'ScreenSize');
fP   = figure;
fD   = figure;

% Locate pattern on second
scP = get(0,'MonitorPositions'); 
pos1    = get(fP,'Position');
%pos1(1) = ;
pos1 = [scSz(3), scP(1,4)-scP(2,4), (scP(2,3) - scP(1,3)), scP(1,4)+abs(scP(1,4)-scP(2,4))];
%pos1(3:4) = ;
set(fP,'MenuBar','None');
set(fP,'ToolBar','None');
WindowAPI(fP,'Clip','True');
set(fP,'Position',pos1);

idx = 0;
while 1
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
    idx        = idx + 1;
    dummyC     = binornd(1,0.5,pos1(4),pos1(3));
    P(:,:,1,idx) = dummyC;
    P(:,:,2,idx) = dummyC;
    P(:,:,3,idx) = dummyC;
    figure(fP)
    imshow(P(:,:,:,idx),'Border','tight');
    set(fP,'MenuBar','None');
    set(fP,'ToolBar','None');
    WindowAPI(fP,'Clip','True');
    WindowAPI(fP,'Maximize');
    set(fP,'Position',pos1);
    pause(5)
    C(:,:,:,idx) = getsnapshot(vid1);
    %figure(fD)
    %subplot(1,3,1);
    %imshow(C)
    %subplot(1,3,2);
    %imagesc(rgb2gray(C));
    %subplot(1,3,3);
    %imshow(P);
    %drawnow
    %pause(0.1)
end