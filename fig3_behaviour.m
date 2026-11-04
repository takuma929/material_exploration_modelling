% fig3_behaviour
% Figure 3: the observers' metal responses (Experiment 3 of Nomachi et al., 2026).
%
% Four square panels across the two-column width: (a) proportion of metal
% responses against physical metallicity by environment, (b) by shape (the
% three exploration conditions), (c) by condition, including the condition
% without exploration; (d) viewing duration by condition.
% Output: figs/fig3.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();

T = readtable(fullfile(project_root, 'data', 'trials.csv'), 'TextType', 'string');
T.participant = string(T.participant);
T.dur_s = (T.t_last_ms - T.t_first_ms) / 1000;
% the static block (trial IDs 1-40, no exploration; python/import_static_block.py)
Ts = readtable(fullfile(project_root, 'data', 'static_trials.csv'), 'TextType', 'string');
Ts.participant = string(Ts.participant);
keep = {'participant', 'trial', 'env', 'geom', 'metal', 'resp', 'modality', 'dur_s'};
TA = [Ts(:, keep); T(:, keep)];       % all four conditions
levels = [0 0.25 0.5 0.75 1];

W = figstyle.WIDTH_2COL; H = 5.0;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);

%% four square panels in a row
pw = 3.2; top = 0.7; cols_x = 1.3 + (0:3) * (pw + 1.15);   % smaller panels, wider gaps
ax_a = ax_cm(cols_x(1), top, pw, pw);
h = local_curves(ax_a, T, T.env, col.envs, col.env, levels, 'resp', {'o', 's', '^', 'd'});
figstyle.set_label(ax_a, 'y', 'Proportion of metal responses');
% key with a thumbnail of each environment (top left, where the panel is empty)
ex_dir = fullfile(project_root, 'data', 'example_frames');
tw = 0.34; mk = {'o', 's', '^', 'd'};
d2x = @(cm) -0.08 + 1.16 * cm / pw; d2y = @(cm) 1 - cm / pw;    % cm offsets inside the panel -> data units
for ie = 1:4
    l = 0.1; t = 0.08 + (ie - 1) * (tw + 0.06);
    axi = axes('Parent', fig, 'Units', 'normalized', 'Position', [(cols_x(1) + l) / W, (H - top - t - tw) / H, tw / W, tw / H]);
    image(axi, imread(fullfile(ex_dir, sprintf('view_%s_Bumpy_m1.png', col.envs{ie})))); axis(axi, 'image', 'off');
    plot(ax_a, d2x(l + tw + 0.2), d2y(t + tw / 2), mk{ie}, 'MarkerSize', figstyle.MS - 2, 'MarkerFaceColor', col.env(ie, :), 'MarkerEdgeColor', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
    text(ax_a, d2x(l + tw + 0.36), d2y(t + tw / 2), col.envs{ie}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'middle');
end

ax_b = ax_cm(cols_x(2), top, pw, pw);
h = local_curves(ax_b, T, T.geom, col.geoms, col.geom, levels, 'resp');
for ig = 1:2
    l = 0.1; t = 0.08 + (ig - 1) * (tw + 0.06);
    axi = axes('Parent', fig, 'Units', 'normalized', 'Position', [(cols_x(2) + l) / W, (H - top - t - tw) / H, tw / W, tw / H]);
    image(axi, imread(fullfile(ex_dir, sprintf('ctx_Sky_%s_m1.png', col.geoms{ig})))); axis(axi, 'image', 'off');
    plot(ax_b, d2x(l + tw + 0.2), d2y(t + tw / 2), 'o', 'MarkerSize', figstyle.MS - 2, 'MarkerFaceColor', col.geom(ig, :), 'MarkerEdgeColor', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
    text(ax_b, d2x(l + tw + 0.36), d2y(t + tw / 2), col.geoms{ig}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'middle');
end

%% (c) by condition: the condition without exploration against the three exploration conditions
ax_c = ax_cm(cols_x(3), top, pw, pw);
h = local_curves(ax_c, TA, TA.modality, col.blocks, col.block, levels, 'resp');
lg = legend(ax_c, h, col.block_labels, 'Location', 'northwest', 'Box', 'off', 'FontSize', figstyle.FS_LEGEND);
for k = 1:4
    sel = TA.modality == col.blocks{k};
    fprintf('  %-6s P(metal) by level: %s\n', col.blocks{k}, mat2str(round(arrayfun(@(l) mean(TA.resp(sel & TA.metal == l)), levels), 2)));
end

%% (d) viewing duration by condition
ax_d = ax_cm(cols_x(4), top, pw, pw);
h = local_curves(ax_d, TA, TA.modality, col.blocks, col.block, levels, 'dur_s');
figstyle.set_ticks(ax_d, 'y', 0:2:10, '%.0f'); ylim(ax_d, [0 10]);
figstyle.set_label(ax_d, 'y', 'Viewing duration [s]');

figstyle.panel_labels(fig, [ax_a ax_b ax_c ax_d], {'(a)', '(b)', '(c)', '(d)'});
figstyle.save(fig, fullfile(project_root, 'figs', 'fig3'));
close(fig);

function h = local_curves(ax, T, groupvar, keys, cols, levels, yvar, markers)
if nargin < 8, markers = repmat({'o'}, 1, numel(keys)); end
h = gobjects(numel(keys), 1);
for k = 1:numel(keys)
    sel = groupvar == keys{k};
    mu = zeros(size(levels)); se = mu;
    for il = 1:numel(levels)
        s = T(sel & T.metal == levels(il), :);
        per = groupsummary(s, 'participant', 'mean', yvar);
        v = per.(['mean_' yvar]);
        mu(il) = mean(v); se(il) = std(v) / sqrt(numel(v));
    end
    h(k) = figstyle.plot_mean(ax, levels, mu, se, cols(k, :), [], [], markers{k});
end
xlim(ax, [-0.08 1.08]);
figstyle.set_ticks(ax, 'x', levels, '%.2f');
figstyle.set_label(ax, 'x', 'Physical metallicity');
if strcmp(yvar, 'resp'); ylim(ax, [0 1]); figstyle.set_ticks(ax, 'y', 0:0.2:1, '%.1f'); end
end
