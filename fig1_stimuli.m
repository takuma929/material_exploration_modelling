% fig1_stimuli
% Figure 1: what the observers saw, and what the models were given.
%
% (a) the patch given to the models (the object plus a margin of its surround)
%     at the five levels of physical metallicity, for the two shapes, Sky;
% (b) the first-person view of one observer in each of the four illumination
%     environments (720 x 720 px screenshots), with that patch outlined;
% (c) the patches for the two shapes at the two ends of physical metallicity
%     in every environment, unnatural on the left and natural on the right.
% The panels follow the order in which the text introduces them.
% Output: figs/fig1.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
ex_dir = fullfile(project_root, 'data', 'example_frames');
B = readtable(fullfile(ex_dir, 'boxes.csv'), 'TextType', 'string');

W = figstyle.WIDTH_2COL; H = 12.9;
fig = figstyle.figure(2, H);
img_cm = @(l, t, w, h) axes('Parent', fig, 'Units', 'normalized', 'Position', [l / W, (H - t - h) / H, w / W, h / H]);
lab = @(ax, x, y, s, varargin) text(ax, x, y, s, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, varargin{:});

%% (b) full first-person views, one per environment (bumpy, physical metallicity 1)
vw = 4.0; gap = 0.35; left0 = 0.6; top0 = 3.55;
for ie = 1:4
    ax = img_cm(left0 + (ie - 1) * (vw + gap), top0, vw, vw);
    image(ax, imread(fullfile(ex_dir, sprintf('view_%s_Bumpy_m1.png', col.envs{ie})))); axis(ax, 'image', 'off');
    b = B(B.env == col.envs{ie} & B.geom == "Bumpy" & B.metal == 1, :);
    rectangle(ax, 'Position', [b.left b.top b.side b.side], 'EdgeColor', col.box, 'LineWidth', 0.8);
    lab(ax, 360, -12, col.envs{ie}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', figstyle.FS_LABEL);
end

%% (a) physical metallicity: one row of patches, both shapes, Sky
lv = [0 0.25 0.5 0.75 1]; lw_ = 1.55; lg_ = 0.08; top_l = 1.0; left_l = 0.6;
for ig = 1:2
    for il = 1:5
        c = (ig - 1) * 5 + il;
        ax = img_cm(left_l + (c - 1) * (lw_ + lg_) + (ig - 1) * 0.5, top_l, lw_, lw_);
        image(ax, imread(fullfile(ex_dir, sprintf('levels_Sky_%s_m%.2f.png', col.geoms{ig}, lv(il))))); axis(ax, 'image', 'off');
        lab(ax, 112, -6, sprintf('%g', lv(il)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        if il == 3
            lab(ax, 112, -40, sprintf('%s, physical metallicity', col.geoms{ig}), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', figstyle.GRAY_TEXT);
        end
    end
end

%% (c) model input patches: two columns (unnatural | natural environments), rows = environment, columns = shape x material
mats = {'glass', 'metal'};
pw = 1.8; pg = 0.08; top_b = 8.9; left_b = 1.0; colw = 4 * pw + 3 * pg; colgap = 1.3;
groups = {{'Check', 'Gray'}, {'Sky', 'Studio'}}; gnames = {'Unnatural environments', 'Natural environments'};
for grp = 1:2
    left_g = left_b + (grp - 1) * (colw + colgap);
    for ie = 1:2
        env = groups{grp}{ie};
        for ig = 1:2
            for im = 1:2
                c = (ig - 1) * 2 + im;
                ax = img_cm(left_g + (c - 1) * (pw + pg), top_b + (ie - 1) * (pw + pg), pw, pw);
                image(ax, imread(fullfile(ex_dir, sprintf('ctx_%s_%s_m%d.png', env, col.geoms{ig}, im - 1)))); axis(ax, 'image', 'off');
                if ie == 1
                    lab(ax, 112, -6, sprintf('%s, %s', col.geoms{ig}, mats{im}), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                end
                if c == 1
                    lab(ax, -8, 112, env, 'HorizontalAlignment', 'right');
                end
                if ie == 1 && c == 2
                    lab(ax, 224 + 0.5 * pg / pw * 224, -40, gnames{grp}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', figstyle.GRAY_TEXT);
                end
            end
        end
    end
end

%% panel labels
ov = figstyle.overlay(fig);
pl = @(x, y, s) text(ov, x / W, 1 - y / H, s, 'Units', 'normalized', 'FontName', figstyle.FONT, ...
    'FontSize', figstyle.FS_PANEL, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
pl(0.1, top_l - 0.5, '(a)'); pl(0.1, top0 - 0.15, '(b)'); pl(0.1, top_b - 0.75, '(c)');
figstyle.save(fig, fullfile(project_root, 'figs', 'fig1'));
close(fig);
