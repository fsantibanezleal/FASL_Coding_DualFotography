%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Santibáñez Peet 2014
%  fsantibanezleal@ug.uchile.cl

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Preliminar implementation for Compressive Dual Photograhy

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial cleaning 
close all;
clear all;
clc;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Provide paths and specific configuration
f0BasicPaths;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Init frame grabber
start(vid1);
wait(vid1,Inf);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Retrieve the frames and timestamps for each frame.
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
%fD   = figure;
%set(fD,'Visible', 'Off', 'Position',[0 0 1 1])   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Locate pattern on second && validate if second monitor is present
if  numel(get(0, 'MonitorPositions'))/4==1
    scP  = get(0,'MonitorPositions'); 
    pos1 = get(0,'MonitorPositions'); 
elseif numel(get(0, 'MonitorPositions'))/4==2
    scP  = get(0,'MonitorPositions'); 
    pos1 = get(fP,'Position');
    %pos1(1) = ;
    pos1  = [ scSz(3)  , scP(1,4)-scP(2,4),...
             scP(2,3) - scP(1,3),...
             scP(1,4) + abs(scP(1,4)-scP(2,4))];
    %pos1(3:4) = ;
end

%eval(['!' cI 'i_view32.exe ' '/slideshow=' pwd filesep 'Images' filesep ' /hide=all /monitor=2 /pos=(' ...
%    num2str(scP(2,1)) ',' num2str(scP(2,2)) ') /silent /reloadonloop /fs &']);
for idx = 1:20
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
    dummyC       = binornd(1,0.5,pos1(4),pos1(3));
    P(:,:,1,idx) = dummyC;
    P(:,:,2,idx) = dummyC;
    P(:,:,3,idx) = dummyC;

    
    imwrite(P(:,:,3,idx),'pattern.png');
    
    
    eval(['!' cI 'i_view32.exe ' 'pattern.png' ' /one /hide=all /monitor=2 /pos=(' ...
    num2str(scP(2,1)) ',' num2str(scP(2,2)) ') /silent /fs &']);

    pause(1)
    
    gSNP = imresize(YUY2toRGB(getsnapshot(vid1)),[pos1(4) pos1(3)]);
    imwrite(gSNP,'dummy.png');
        
    C(:,:,:,idx) = gSNP;
    figure(fP)
    subplot(2,2,1);
    imshow(gSNP)
    subplot(2,2,2);
    imshow(rgb2gray(gSNP));
    subplot(2,2,[3 4]);
    imshow(dummyC);
    set(fP, 'Position', get(0,'Screensize')); % Maximize figure.
    drawnow
    %pause(0.1)
end
eval(['!' cI 'i_view32.exe ' '/killmesoftly &']);