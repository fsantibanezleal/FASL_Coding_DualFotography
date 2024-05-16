function uTest_WindowAPI(doSpeed)
% Automatic test: WindowAPI
% This is a routine for automatic testing. It is not needed for processing and
% can be deleted or moved to a folder, where it does not bother.
%
% The methods of CmdWinTool depend on undocumented feature of Matlab. If a
% specific action fails on your platform, modify CmdWinTool. I appreciate
% reports or problems by email.
%
% uTest_WindowAPI(doSpeed)
% INPUT:
%   doSpeed: If this is 0 or FALSE, a faster test is performed. For other values
%            or if omitted each actipon is shown for 0.5 seconds.
% OUTPUT:
%   On failure the test stops with an error.
%
% Tested: Matlab 6.5, 7.7, 7.8, 7.13, WinXP/32, Win7/64
% Author: Jan Simon, Heidelberg, (C) 2009-2011 matlab.THISYEAR(a)nMINUSsimon.de

% $JRev: R-e V:004 Sum:+9poaLgXgrKr Date:09-Sep-2011 17:34:12 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $File: Tools\UnitTests_\uTest_WindowAPI.m $
% History:
% 001: 26-Jun-2011 15:20, First version.
% 004: 24-Jul-2011 18:23, LockCursor.

% Initialize: ==================================================================
FuncName = mfilename;
ErrID    = ['JSimon:', FuncName, ':Crash'];

LF = char(10);

if nargin == 0
   doSpeed = true;
end
if doSpeed
   Delay = 0.5;
else
   Delay = 0.02;
end

% Do the work: =================================================================
% Hello:
disp(['==== Test WindowAPI:  ', datestr(now, 0), LF, ...
   '  Version: ', which('CmdWinTool'), LF]);

% Create figure on 1st monitor:
MonitorIndex = 0;
FigH = figure('Color', [100, 200, 0] ./ 255, ...
   'NumberTitle', 'off', ...
   'Name',        'Test WindowAPI', ...
   'Renderer',    'Painters');
FigPos = get(FigH, 'Position');
BtnPos = [60, 60, 120, 40];
ButtonH = uicontrol('Style', 'Togglebutton', 'String', '', ...
   'FontSize', 16, ...
   'Position', BtnPos);
AxesH = axes;
sphere;
set(AxesH, 'visible', 'off', 'CameraViewAngle', 30);

ready = false;
while not(ready)
   MonitorIndex = MonitorIndex + 1;
   fprintf('== Monitor %d\n', MonitorIndex);
   set(ButtonH, 'String', sprintf('Monitor %d', MonitorIndex));
   
   try
      WindowAPI(FigH, 'TopMost');
      pause(Delay);
      WindowAPI(FigH, 'NoTopMost');
      pause(Delay);
      fprintf('  ok: TopMost, NoTopMost\n');
   catch
      error(ErrID, '*** %s: TopMost/NoTopMost crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'Front');
      pause(Delay);
      fprintf('  ok: Front\n');
   catch
      error(ErrID, '*** %s: Front crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'Minimize');
      Status = WindowAPI(FigH, 'GetStatus');
      if strcmpi(Status, 'minimized') == 0
         error('GetStatus(minimized) failed');
      end
      pause(Delay);
      
      WindowAPI(FigH, 'Maximize');
      Status = WindowAPI(FigH, 'GetStatus');
      if strcmpi(Status, 'maximized') == 0
         error('GetStatus(maximized) failed');
      end
      pause(Delay);
      
      WindowAPI(FigH, 'Restore');
      Status = WindowAPI(FigH, 'GetStatus');
      pause(Delay);
      if strcmpi(Status, 'restored') == 0
         error('GetStatus(restored) failed');
      end
      fprintf('  ok: Minimize/Maximize/Restore\n');
   catch
      error(ErrID, ...
         '*** %s: Minimize/Maximize/Restore crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'XMax');
      pause(Delay);
      WindowAPI(FigH, 'Position', FigPos);
      WindowAPI(FigH, 'YMax');
      pause(Delay);
      WindowAPI(FigH, 'Position', FigPos);
      fprintf('  ok: XMax, YMax\n');
   catch
      error(ErrID, '*** %s: XMax/YMax crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'Position', [100, 102, 600, 202]);
      pause(Delay);
      WindowAPI(FigH, 'Position', 'work');
      pause(Delay);
      WindowAPI(FigH, 'Position', 'full');
      pause(Delay);
      drawnow;
      pause(Delay);
      WindowAPI(FigH, 'Position', FigPos);
      
      fprintf('  ok: Position\n');
   catch
      error(ErrID, '*** %s: Position crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'OuterPosition', [100, 102, 200, 202]);
      pause(Delay);
      WindowAPI(FigH, 'OuterPosition', 'work');
      pause(Delay);
      WindowAPI(FigH, 'OuterPosition', 'full');
      pause(Delay);
      WindowAPI(FigH, 'Position', FigPos);
      
      fprintf('  ok: OuterPosition\n');
   catch
      error(ErrID, '*** %s: OuterPosition crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'Flash');
      pause(Delay);
      fprintf('  ok: Flash\n');
   catch
      error(ErrID, '*** %s: Flash crashed:\n%s', FuncName, lasterr);
   end
   
   try
      for dAlpha = linspace(1, 0, 10)
         WindowAPI(FigH, 'Alpha', dAlpha);
         pause(0.1);
      end
      pause(Delay);
      fprintf('  ok: Alpha\n');
      
      set(FigH, 'Color', [1, 0, 1]);
      for dAlpha = linspace(0, 1, 10)
         WindowAPI(FigH, 'Alpha', dAlpha, [255, 0, 255]);
         pause(0.02);
      end
      pause(Delay);
      WindowAPI(FigH, 'Opaque');
      fprintf('  ok: Alpha and StencilRGB\n');
   catch
      error(ErrID, '*** %s: Alpha crashed:\n%s', FuncName, lasterr);
   end
   
   try
      set(FigH, 'Color', [100, 200, 0] ./ 255);
      IniPos  = [1, 1, FigPos(3:4)];
      DiffPos = (BtnPos - IniPos) / 10;
      for i = 1:10
         WindowAPI(FigH, 'Clip', IniPos + i * DiffPos);
         pause(0.05);
      end
      pause(Delay);
      WindowAPI(FigH, 'Clip', false);
      fprintf('  ok: Clip\n');
   catch
      error(ErrID, '*** %s: Clip crashed:\n%s', FuncName, lasterr);
   end
   
   try
      Monitor = WindowAPI(FigH, 'Monitor');
      fprintf('  ok: Monitor:\n');
      disp(Monitor);
   catch
      error(ErrID, '*** %s: Monitor crashed:\n%s', FuncName, lasterr);
   end
   
   try  % Set window position on the current monitor:
      Pos1 = WindowAPI(FigH, 'Position');
      WindowAPI(FigH, 'Position', Pos1.Position);
      Pos2 = WindowAPI(FigH, 'Position');
      
      if isequal(Pos1, Pos2) == 0
         error('Get/Set Position failed');
      end
      if Pos1.MonitorIndex == 1
         if isequal(get(FigH, 'Position'), Pos2.Position) == 0
            error('Get/Set Position failed on primary monitor');
         end
      end
      
      fprintf('  ok: Get/Set Position\n');
   catch
      error(ErrID, '*** %s: Monitor crashed:\n%s', FuncName, lasterr);
   end
   
   try  % Set outer window position on the current monitor:
      Pos1 = WindowAPI(FigH, 'OuterPosition');
      WindowAPI(FigH, 'OuterPosition', Pos1.Position);
      Pos2 = WindowAPI(FigH, 'OuterPosition');
      
      if isequal(Pos1, Pos2) == 0
         error('Get/Set OuterPosition failed');
      end
      if Pos1.MonitorIndex == 1
         if isequal(get(FigH, 'OuterPosition'), Pos2.Position) == 0
            error('Get/Set OuterPosition failed on primary monitor');
         end
      end
      fprintf('  ok: Get/Set OuterPosition\n');
   catch
      error(ErrID, '*** %s: Monitor crashed:\n%s', FuncName, lasterr);
   end
   
   try
      WindowAPI(FigH, 'LockCursor', 1);
      pause(0.1);
      WindowAPI(FigH, 'LockCursor', 0);
      pause(0.1);
      WindowAPI(FigH, 'LockCursor', [10, 10, 200, 100]);
      pause(0.1);
      WindowAPI(FigH, 'LockCursor');
      pause(0.1);
      WindowAPI('UnlockCursor');
   catch
      error(ErrID, '*** %s: LockCursor crashed:\n%s', FuncName, lasterr);
   end
   
   % Move figure to next screen:
   MonitorIndex = MonitorIndex + 1;
   try
      WindowAPI(FigH, 'Position', FigPos, MonitorIndex);
      
      % Keep window visible if the monitors have different sizes:
      WindowAPI(FigH, 'ToScreen');
      Pos    = WindowAPI(FigH, 'Position');
      FigPos = Pos.Position;
      
      % If a not existing monitor is chosen, the figure is moved to the primary
      % monitor:
      Monitor = WindowAPI(FigH, 'Monitor');
      if Monitor.MonitorIndex == 1
         ready = true;
      end
   catch
      error(ErrID, '*** %s: Monitor crashed:\n%s', FuncName, lasterr);
   end
   fprintf('\n');
end  % while not(ready)

% Goodbye:
delete(FigH);
fprintf('WindowAPI passed the tests.\n');

return;
