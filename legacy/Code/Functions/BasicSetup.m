close all;
clear all;
clc;
% See available adapters for video

adV    = imaqhwinfo;
ad1    = adV.InstalledAdaptors{4};
wInfoV = imaqhwinfo(ad1);
numD   = size(wInfoV.DeviceIDs,2);
devIDU = 2;
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

% for idxF =1:size(frames,4)
%     imshow(frames(:,:,:,idxF))
%     pause(0.1)
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recovery by image
% video1 = imaqhwinfo('winvideo',devIDU);
set(0,'Units','pixels') 
scSz = get(0,'ScreenSize');
fP   = figure;
fD   = figure;
%set(fD,'Visible', 'Off', 'Position',[0 0 1 1])   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Locate pattern on second
scP = get(0,'MonitorPositions'); 
pos1    = get(fP,'Position');
%pos1(1) = ;
pos1 = [ scSz(3)  , scP(1,4)-scP(2,4),...
         scP(2,3) - scP(1,3),...
         scP(1,4) + abs(scP(1,4)-scP(2,4))];
%pos1(3:4) = ;

for idx = 1:2
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
    dummyC     = binornd(1,0.5,pos1(4),pos1(3));
    P(:,:,1,idx) = dummyC;
    P(:,:,2,idx) = dummyC;
    P(:,:,3,idx) = dummyC;
    figure(fP)
    imshow(P(:,:,:,idx));%,'Border','tight');  
    %I = imread('pout.tif');
    %imshow(I);
    set(fP,'Position',pos1); 
    
set(fP,'windowstyle','modal');
set(fP,'MenuBar','None');
set(fP,'ToolBar','None');

%set(fP,'OuterPosition', pos1); 
%set(fP,'position',pos1); 
%set(fP,'Units','normal', 'outerposition',[0 0 1 1])    
    WindowAPI(fP, 'Clip', true);
    set(fP,'Position',pos1); 
    %WindowAPI(fP, 'position', 'full');
    posDummy     = get(fP,'Position');
        %set(fP,'Visible', 'Off', 'Position',[0 0 1 1]) 
        set(gca,'Visible', 'Off', 'Position',[0 0 1 1])       
        set(gca,'units','pixels')
        pos2     = get(gca,'Position');
        %pos2(2)  = 1;
        %set(gca,'Position',pos2);
        %set(gca,'units','normalized','position',[0 0 1 1])
 
%     pause(5)
%     C(:,:,:,idx) = getsnapshot(vid1);
%     figure(fD)
%     subplot(1,3,1);
%     imshow(C(:,:,:,idx))
%     subplot(1,3,2);
%     imshow(rgb2gray(C(:,:,:,idx)));
%     subplot(1,3,3);
%     imshow(P(:,:,1,idx));
%     %drawnow
    pause(0.1)
end