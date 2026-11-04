function c = project_colors()
% PROJECT_COLORS  Colors shared by every figure: one blue ramp plus grays.
%   Every ordered or grouped factor (exploration condition, environment,
%   shape, physical metallicity, time window) takes steps of figstyle.SEQ_BLUE,
%   light to dark; reference predictors are gray; the own-video model is the
%   dark blue accent used throughout; one red accent marks the crop box drawn
%   on images and one magenta accent the statistics of Figure 5 in Figure 6.
B = figstyle.SEQ_BLUE;
c.ramp = B;
c.accent = B(4, :);                        % the own-video model (the statistics of the observer's own video)

% the four conditions, in the order of the figures and the text: the three
% exploration conditions, then the condition without exploration (gray)
c.static = [0.62 0.62 0.62];
c.block = [B([1 3 5], :); c.static];
c.blocks = {'head', 'hand', 'both', 'static'};
c.block_labels = {'Head only', 'Hand only', 'Head + hand', 'No exploration'};

c.metal_param = [0.72 0.72 0.72];          % physical metallicity as a predictor
c.stats       = c.accent;                  % the own-video model
c.null        = [0.85 0.85 0.85];          % permutation null band
c.glass_dir   = B(1, :);                   % a statistic whose higher values go with "glass"
c.metal_dir   = B(4, :);                   % a statistic whose higher values go with "metal"
c.box         = [220 68 92] / 255;         % crop box drawn on images
c.highlight   = [0.85 0.15 0.65];          % magenta: the 16 statistics of Figure 5, marked in Figure 6

c.env = B([1 2 4 5], :);                   % Check, Gray, Sky, Studio
c.envs = {'Check', 'Gray', 'Sky', 'Studio'};
c.geom = B([5 2], :);                      % Bumpy, Smooth
c.geoms = {'Bumpy', 'Smooth'};
end
