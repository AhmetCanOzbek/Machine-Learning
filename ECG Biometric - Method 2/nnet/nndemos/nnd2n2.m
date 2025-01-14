function nnd2n2(cmd,arg1,arg2,arg3)
%NND2N2 Two-input neuron demonstration.
%
%  Control the output of a two-input neuron by adjusting its
%  two inputs, two weights, bias, and transfer function.

% $Revision: 1.7.2.5 $
% Copyright 1994-2011 Martin T. Hagan
% First Version, 8-31-95.

%==================================================================

% CONSTANTS
me = 'nnd2n2';
max_t = 0.5;
w_max = 2;
p_max = 1;
n_max = 6;
a_max = 6;

% DEFAULTS
if nargin == 0, cmd = ''; else cmd = lower(cmd); end

% FIND WINDOW IF IT EXISTS
fig = nndfgflg(me);
if ~isempty(fig) && isempty(get(fig,'children')), fig = []; end
  
% GET WINDOW DATA IF IT EXISTS
if ~isempty(fig)
  H = get(fig,'userdata');
  fig_axis = H(1);            % window axis
  desc_text = H(2);           % handle to first line of text sequence
  meters = H(3:9);            % input and output meters (axes)
  indicators = H(10:16);      % input and output indicators (patches)
  w_ptr = H(17);              % pointer to weight vector
  b_ptr = H(18);              % pointer to bias value
  f_ptr = H(19);              % pointer to transfer function
  p_ptr = H(20);              % pointer to input vector
  f_menu = H(21);             % transfer function menu
  f_text = H(22);             % neuron heading text
  f_text2 = H(23);            % neuron function text
end

%==================================================================
% Activate the window.
%
% ME() or ME('')
%==================================================================

if strcmp(cmd,'')
  if ~isempty(fig)
    figure(fig)
    set(fig,'visible','on')
  else
    feval(me,'init')
  end

%==================================================================
% Close the window.
%
% ME() or ME('')
%==================================================================

elseif strcmp(cmd,'close') & ~isempty(fig)
  delete(fig)

%==================================================================
% Initialize the window.
%
% ME('init')
%==================================================================

elseif strcmp(cmd,'init') & isempty(fig)

  % CHECK FOR TRANSFER FUNCTIONS
  if ~nnfexist(me), return, end

  % CONSTANTS
  w = [0.5 -0.5]*w_max;
  b = [0];
  p = [0; 0];
  f = 'purelin';
  n = w*p+b;
  a = feval(f,n);
  title_str = 'Neuron Model Demonstration';
  chapter_str = 'Chapter 2';

  % NEW DEMO FIGURE
  fig = nndemof2(me,'DESIGN','Two-Input Neuron','','Chapter 2');
  set(fig, ...
    'windowbuttondownfcn',nncallbk(me,'down'), ...
    ... %'Backing_Store','off',...
    'nextplot','add');
  H = get(fig,'userdata');
  fig_axis = H(1);
  desc_text = H(2);

  % ICON
  nndicon(2,458,363,'shadow')

  % NEURON DIAGRAM
  if ~isa(gca,'double')
    set(gca,'SortMethod','ChildOrder')
  end
  x = 60;
  y = 230;
  plot(x+[0 100 0],y-[0 50 100],...
   'linewidth',4,...
   'color',nnred);
  plot(x+[100 100],y-[49 112],...
   'linewidth',4,...
   'color',nnred);
  nndicon(100,x+100,y-50)
  plot(x+[125 136],y-[50 50],...
   'linewidth',4,...
   'color',nnred);
  plot(x+[160 185],y-[50 50],...
   'linewidth',4,...
   'color',nnred);
  plot(x+[200 250],y-[50 50],...
   'linewidth',4,...
   'color',nnred);
  plot(x+[240 250 240],y-[40 50 60],...
   'linewidth',4,...
   'color',nnred);
  nndicon(101,x+200,y-50)

  deg = pi/180;
  angle = [0:5:90]*deg;
  xc = cos(angle)*10;
  yc = sin(angle)*10;

  plot(x-20-xc,y+70+yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+[-20 0],y+[80 80],...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+xc,y+70+yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x-20-xc,y-160-yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+[-20 0],y-[170 170],...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+xc,y-160-yc,...
   'linewidth',3,...
   'color',nndkblue);
  text(x-10,y+95,'Input',...
    'color',nndkblue,...
    'fontweight','bold',...
    'horiz','center');

  plot(x+30-xc,y+70+yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+[30 280],y+[80 80],...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+280+xc,y+70+yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+30-xc,y-160-yc,...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+[30 280],y-[170 170],...
   'linewidth',3,...
   'color',nndkblue);
  plot(x+280+xc,y-160-yc,...
   'linewidth',3,...
   'color',nndkblue);
  f_text = text(x+155,y+95,'Linear Neuron',...
    'color',nndkblue,...
    'fontweight','bold',...
    'horiz','center',...
    'CreateFcn','');
  func_str = sprintf('a = %s(w*p+b) = %g',f,a);
  f_text2 = text(x+155,y-190,'a = purelin(w*p+b)',...
    'color',nndkblue,...
    'fontweight','bold',...
    'horiz','center',...
    'CreateFcn','');

  text(x+155,y+45,'F',...
    'fontname','helvetica',...
    'fontweight','bold',...
    'color',nndkblue,...
    'horiz','center',...
    'fontsize',12)

  % SIGNALS
  p1_axis = nnsfo('a2','p(1)','','');
  set(p1_axis, ...
    'units','points',...
    'position',[x-20 y-35 20 70],...
    'color',nnltyell,...
    'xlim',[-0.3 1.3],...
    'xtick',[],...
    'ylim',[-1.3 1.3]*p_max,...
    'ytick',[-1 -0.5 0 0.5 1]*p_max,...
    'yticklabel',char(num2str(-p_max),'','0','',num2str(p_max)))
  p1_ind = fill([0 1 1],[0 0.2 -0.2]*p_max+p(1),nnred,...
    'edgecolor',nndkblue,...
    'CreateFcn','');
  p2_axis = nnsfo('a2','p(2)','','');
  set(p2_axis, ...
    'units','points',...
    'position',[x-20 y-135 20 70],...
    'color',nnltyell,...
    'xlim',[-0.3 1.3],...
    'xtick',[],...
    'ylim',[-1.3 1.3]*p_max,...
    'ytick',[-1 -0.5 0 0.5 1]*p_max,...
    'yticklabel',char(num2str(-p_max),'','0','',num2str(p_max)))
  p2_ind = fill([0 1 1],[0 0.2 -0.2]*p_max+p(2),nnred,...
    'edgecolor',nndkblue,...
    'CreateFcn','');
  n_axis = nnsfo('a2','n','','');
  set(n_axis, ...
    'units','points',...
    'position',[x+140 y-85 20 70],...
    'color',nnltyell,...
    'xlim',[-0.3 1.3],...
    'xtick',[],...
    'ylim',[-1.3 1.3]*n_max,...
    'ytick',-n_max:2:n_max,...
    'yticklabel',char(num2str(-n_max),'','','','','',num2str(n_max)))
  n_ind = fill([0 1 1],[0 0.2 -0.2]*n_max+n,nndkblue,...
    'edgecolor','none',...
    'CreateFcn','');
  a_axis = nnsfo('a2','a','','');
  set(a_axis, ...
    'units','points',...
    'position',[x+270 y-120 20 140],...
    'color',nnltyell,...
    'xlim',[-0.3 1.3],...
    'xtick',[],...
    'ylim',[-1.15 1.15]*a_max,...
    'ytick',[-a_max:1:a_max],...
    'yticklabel',...
       ['-6';'  ';'-4';'  ';'-2';'  ';'0 ';'  ';'2 ';'  ';'4 ';'  ';'6 '])
  a_ind = fill([0 1 1],[0 0.2 -0.2]*a_max/2+a,nndkblue,...
    'edgecolor','none',...
    'CreateFcn','');
  bi_axis = nnsfo('a2','','','');
  set(bi_axis, ...
    'units','points',...
    'position',[x+90 y-135 20 20],...
    'color',nnltyell,...
    'xlim',[-1 1],...
    'xtick',[],...
    'ylim',[-1 1],...
    'ytick',[])
  bi_ind = text(0,0,'1',...
    'color',nndkblue,...
    'fontweight','bold',...
    'horiz','center');

  % PARAMETERS
  w1_axis = nnsfo('a2','w(1,1)','','');
  set(w1_axis, ...
    'units','points',...
    'position',[x+10 y+15 70 20],...
    'color',nnmdgray,...
    'ylim',[-0.3 1.3],...
    'ytick',[],...
    'xlim',[-1.3 1.3]*w_max,...
    'xtick',[-1 -0.5 0 0.5 1]*w_max,...
    'xticklabel',char(num2str(-w_max),'','0','',num2str(w_max)))
  w1_ind = fill([0 0.2 -0.2]*w_max+w(1),[0 1 1],nnred,...
    'edgecolor',nndkblue,...
    'CreateFcn','');
  w2_axis = nnsfo('a2','w(1,2)','','');
  set(w2_axis, ...
    'units','points',...
    'position',[x+10 y-135 70 20],...
    'color',nnmdgray,...
    'ylim',[-0.3 1.3],...
    'ytick',[],...
    'xlim',[-1.3 1.3]*w_max,...
    'xtick',[-1 -0.5 0 0.5 1]*w_max,...
    'xticklabel',char(num2str(-w_max),'','0','',num2str(w_max)))
  w2_ind = fill([0 0.2 -0.2]*w_max+w(2),[0 1 1],nnred,...
    'edgecolor',nndkblue,...
    'CreateFcn','');
  b_axis = nnsfo('a2','b','','');
  set(b_axis, ...
    'units','points',...
    'position',[x+120 y-135 70 20],...
    'color',nnmdgray,...
    'ylim',[-0.3 1.3],...
    'ytick',[],...
    'xlim',[-1.3 1.3]*w_max,...
    'xtick',[-1 -0.5 0 0.5 1]*w_max,...
    'xticklabel',char(num2str(-w_max),'','0','',num2str(w_max)))
  b_ind = fill([0 0.2 -0.2]*w_max+b,[0 1 1],nnred,...
    'edgecolor',nndkblue,...
    'CreateFcn','');
  f_menu = uicontrol(...
    'units','points',...
    'position',[x+115 y+15, 80 20],...
    'style','popupmenu',...
    'string','Hardlim|Hardlims|Purelin|Satlin|Satlins|Logsig|Tansig',...
    'background',nnmdgray,...
    'callback',[me '(''function'')'],...
    'value',3);

  % SLIDE BARS
  drawnow % Let everything else appear before buttons 

  % BUTTONS
  uicontrol(...
    'units','points',...
    'position',[400 110 60 20],...
    'string','Contents',...
    'callback','nndtoc')
  uicontrol(...
    'units','points',...
    'position',[400 75 60 20],...
    'string','Close',...
    'callback',[me '(''close'')'])

  % DATA POINTERS
  meters = [p1_axis p2_axis w1_axis w2_axis b_axis n_axis a_axis];
  indicators = [p1_ind p2_ind w1_ind w2_ind b_ind n_ind a_ind];
  w_ptr = uicontrol('visible','off'); set(w_ptr,'userdata',w);
  b_ptr = uicontrol('visible','off'); set(b_ptr,'userdata',b);
  f_ptr = uicontrol('visible','off'); set(f_ptr,'userdata',f);
  p_ptr = uicontrol('visible','off'); set(p_ptr,'userdata',p);

  % SAVE WINDOW DATA AND LOCK
  H = nndArray({fig_axis desc_text meters indicators w_ptr b_ptr f_ptr p_ptr ...
    f_menu f_text f_text2});
  set(fig,'userdata',H,'nextplot','new')

  % INSTRUCTION TEXT
  feval(me,'instr');

  % LOCK WINDOW
  set(fig,...
   'nextplot','new',...
   'color',nnltgray);

  nnchkfs;

%==================================================================
% Display the instructions.
%
% ME('instr')
%==================================================================

elseif strcmp(cmd,'instr') & ~isempty(fig)
  nnsettxt(desc_text,...
    'Alter the input values',...
    'by clicking & dragging',...
    'the triangle indicators.',...
    '',...
    'Alter the weights and',...
    'bias in the same way.',...
    'Use the menu to pick',...
    'a transfer function.',...
  '',...
  'Pick the transfer',...
  'function with the',...
  'F menu.',...
    '',...
    'The net input and the',...
    'output will respond to',...
    'each change.')
    
%==================================================================
% Respond to mouse down.
%
% ME('down')
%==================================================================

elseif strcmp(cmd,'down') & ~isempty(fig) & (nargin == 1)

  q = 0;
  for i=1:5
    pt = get(meters(i),'currentpoint');
    x = pt(1);
    y = pt(3);

    if (i <= 2)
      if (y >= -1.3*p_max) & (y <= 1.3*p_max) & (x >= 0) & (x <= 1)
        q = i;
        data = 'ydata';
        z_max = p_max;
        z = y;
        hide_color = nnltyell;
        break;
      end
    else
      if (x >= -1.3*w_max) & (x <= 1.3*w_max) & (y >= 0) & (y <= 1)
        q = i;
        data = 'xdata';
        z_max = w_max;
        z = x;
        hide_color = nnmdgray;
        break;
      end
    end
  end

  if (q)
    set(fig,'pointer','crosshair')
    z = min(z_max,max(-z_max,z));
    set(indicators(q),...
      'facecolor',hide_color,...
      'edgecolor',hide_color)
    set(indicators(q),...
      data,[0 0.2 -0.2]*z_max+z,...
      'facecolor',nnred,...
      'edgecolor',nndkblue)
    set(fig,'WindowButtonMotionFcn',[me '(''down'')']);
    set(fig,'WindowButtonUpFcn',[me '(''up'')']);

    % ALTER VARIABLES
    if (q <= 2)
      p = get(p_ptr,'userdata');
      p(q) = z;
      set(p_ptr,'userdata',p);
    elseif (q <= 4)
      w = get(w_ptr,'userdata');
      w(q-2) = z;
      set(w_ptr,'userdata',w);
    else
      set(b_ptr,'userdata',z);
    end

    cmd = 'update';
  else
    set(fig,'pointer','arrow')
  end

%==================================================================
% Respond to mouse up.
%
% ME('up')
%==================================================================

elseif strcmp(cmd,'up') & ~isempty(fig) & (nargin == 1)

  set(fig,...
    'WindowButtonMotionFcn','',...
    'pointer','arrow')

%==================================================================
% Respond to function menu.
%
% ME('function')
%==================================================================

elseif strcmp(cmd,'function') & ~isempty(fig) & (nargin == 1)

  v = get(f_menu,'value');

  if     v == 1, f = 'hardlim';  new_text = 'Hard Limit Neuron';
  elseif v == 2, f = 'hardlims'; new_text = 'Sym. Hard Limit Neuron';
  elseif v == 3, f = 'purelin';  new_text = 'Linear Neuron';
  elseif v == 4, f = 'satlin';   new_text = 'Saturating Linear Neuron';
  elseif v == 5, f = 'satlins';  new_text = 'Sym. Saturating Linear Neuron';
  elseif v == 6, f = 'logsig';   new_text = 'Log Sigmoid Neuron';
  elseif v == 7, f = 'tansig';   new_text = 'Tan Sigmoid Neuron';
  end
  
  set(f_text,...
    'color',nnltgray);
  set(f_text,...
    'string',new_text,...
    'color',nndkblue);
  set(f_text2,...
    'color',nnltgray);
  set(f_text2,...
    'string',['a = ' f '(w*p+b)'],...
    'color',nndkblue)
  set(f_ptr,'userdata',f);
  cmd = 'update';

%==================================================================
end

%==================================================================
% Respond to request to update displays.
%
% ME('update')
%==================================================================

if strcmp(cmd,'update') & ~isempty(fig)
  
  % GET DATA
  w = get(w_ptr,'userdata');
  b = get(b_ptr,'userdata');
  f = get(f_ptr,'userdata');
  p = get(p_ptr,'userdata');

  % UPDATE NET INPUT
  n = w*p+b;
  set(indicators(6),...
    'facecolor',nnltyell)
  set(indicators(6),...
    'ydata',[0 0.2 -0.2]*n_max+n,...
    'facecolor',nndkblue)

  % UPDATE OUTPUT
  n = w*p+b;
  if strcmp(f,'satlin')
    a = (~((n < 0) | (n > 1))).*n + (n > 1);
  elseif strcmp(f,'satlins')
    a = (~((n < -1) | (n > 1))).*n + (n > 1) - (n < -1);
  else
    a = feval(f,n);
  end

  set(indicators(7),...
    'facecolor',nnltyell)
  set(indicators(7),...
    'ydata',[0 0.2 -0.2]*a_max/2+a,...
    'facecolor',nndkblue)
end
