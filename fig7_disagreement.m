% fig7_disagreement
% Figure 7: the disagreement test. Does the video an observer generated
% predict that observer's response beyond the stimulus?
%
% (a) two observers who saw the same trial and responded differently:
%     eight frames of each video (the model's input patches) and the
%     predicted metallicity the own-video model gave each video;
% (b) the AUC over all disagreeing pairs (leftmost) and for the pairs of each
%     illumination environment and of each shape, with the null of each
%     (python/within_trial_breakdown.py);
% (c) the AUC by how much the two videos of a pair differ: quarters of the
%     distance between the two videos in the space of the statistics
%     (SPLIT = 'similarity', python/within_trial_by_similarity.py) or of the
%     absolute difference in predicted metallicity (SPLIT = 'score',
%     python/within_trial_by_difference.py), with the null of each quarter;
% (d) the AUC for the statistics computed over the first 2 s, the first
%     half, the second half and the whole trial, with the null.
% Earlier layouts (with the duration control as a panel; with the AUC of all
% pairs against its null histogram as panel b) are kept in legacy/.
% Output: figs/fig7.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
mdl = fullfile(project_root, 'results', 'models');

M = readtable(fullfile(project_root, 'results', 'summary', 'model_summary.csv'), 'TextType', 'string');
pick = @(fs, C) M(M.feature_set == fs & abs(M.C - C) < 1e-9, :);
rows = [pick("object_relational_flow_first2s", 0.1); pick("object_relational_flow_early_half", 0.1); pick("object_relational_flow_late_half", 0.1); pick("object_relational_flow", 0.1)];
tint = @(f) f * col.accent + (1 - f) * [1 1 1];
cols = [tint(0.3); tint(0.5); tint(0.7); col.accent];
wt = rows.within_trial_auc;
nm = rows.within_trial_null_mean; ns = rows.within_trial_null_sd;
Wb = readtable(fullfile(mdl, 'within_trial_by_observer.csv'));

W = figstyle.WIDTH_2COL; H = 11.3;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);
img_cm = @(l, t, w, h) axes('Parent', fig, 'Units', 'normalized', 'Position', [l / W, (H - t - h) / H, w / W, h / H]);
n_lab = @(ax, x, y, s, va) text(ax, x, y, s, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', figstyle.GRAY_TEXT, 'HorizontalAlignment', 'left', 'VerticalAlignment', va);

%% (a) same trial, two observers, two responses: four of the saved frames of each video
ex_dir = fullfile(project_root, 'data', 'example_frames');
P = readtable(fullfile(ex_dir, 'pair.csv'), 'TextType', 'string');
pg = 0.08; left_a = 3.0; top_a = 0.95; n_show = 8; pw = (W - 0.4 - left_a) / n_show - pg;   % eight frames across the width
roles = {'metal', 'glass'};
for r = 1:2
    fr = dir(fullfile(ex_dir, sprintf('pair_%s_*.png', roles{r})));
    [~, ord] = sort(cellfun(@(n) sscanf(n, ['pair_' roles{r} '_%d_']), {fr.name})); fr = fr(ord);
    fr = fr(round(linspace(1, numel(fr), n_show)));
    for k = 1:numel(fr)
        ax = img_cm(left_a + (k - 1) * (pw + pg), top_a + (r - 1) * (pw + 0.45), pw, pw);
        image(ax, imread(fullfile(ex_dir, fr(k).name))); axis(ax, 'image', 'off');
        t = sscanf(fr(k).name, ['pair_' roles{r} '_%*d_%fs.png']);
        text(ax, 112, -6, sprintf('%.1f s', t), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND);
    end
    p = P(P.role == roles{r}, :);
    obs = p.participant; if ~isnumeric(obs), obs = str2double(obs); end
    text(ax, -(numel(fr) - 1) * (pw + pg) / pw * 224 - 20, 112, ...
        {sprintf('Observer %d', obs), sprintf('said "%s"', roles{r}), sprintf('predicted metallicity %+.1f', p.score)}, ...
        'HorizontalAlignment', 'right', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND);
end
p = P(1, :); parts = split(p.folder, "_");
ov0 = figstyle.overlay(fig);
text(ov0, 0.95 / W, 1 - (top_a - 0.45) / H, sprintf('Same trial: %s, %s, physical metallicity %g, head + hand', parts{2}, lower(parts{3}), str2double(parts{4})), ...
    'Units', 'normalized', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', figstyle.GRAY_TEXT, 'VerticalAlignment', 'bottom');

%% (c) by how much the two videos of a pair differ
SPLIT = 'similarity';
if strcmp(SPLIT, 'similarity')
    Q = readtable(fullfile(mdl, 'within_trial_by_similarity.csv'));
    Qo = readtable(fullfile(mdl, 'within_trial_by_similarity_by_observer.csv'));
else
    Q = readtable(fullfile(mdl, 'within_trial_by_difference.csv'));
    Qo = readtable(fullfile(mdl, 'within_trial_by_difference_by_observer.csv'));
end
top2 = 6.0; h2 = 3.9;
ax_c = ax_cm(8.6, top2, 3.9, h2);
for k = 1:height(Q)
    fill(ax_c, k + 0.42 * [-1 1 1 -1], Q.null_mean(k) + 2 * Q.null_sd(k) * [-1 -1 1 1], col.null, 'EdgeColor', 'none');
end
for k = 1:height(Q)
    bar(ax_c, k, Q.within_trial_auc(k), 0.45, 'FaceColor', col.accent, 'EdgeColor', col.accent, 'LineWidth', 0.4, 'BaseValue', 0.5);
    errorbar(ax_c, k, Q.within_trial_auc(k), std(Qo.(sprintf('q%d', k))) / sqrt(height(Qo)), 'LineStyle', 'none', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_ERR, 'CapSize', 0);
    fprintf('  %s quarter %d (%.2f-%.2f, n %d): AUC %.3f  null %.3f +- %.3f  z %.1f\n', ...
        SPLIT, k, Q.lo(k), Q.hi(k), Q.n(k), Q.within_trial_auc(k), Q.null_mean(k), Q.null_sd(k), Q.z(k));
end
figstyle.ref_line(ax_c, 'h', 0.5, '');
xlim(ax_c, [0.4 4.6]); ylim(ax_c, [0.45 0.75]);
en = char(8211);
if strcmp(SPLIT, 'similarity')
    figstyle.set_ticks(ax_c, 'x', 1:4, {'', '', '', ''}); center_xticklabels(ax_c, {{'Most', 'similar'}, {'2nd', 'quarter'}, {'3rd', 'quarter'}, {'Most', 'different'}});
    text(ax_c, 2.5, 0.45 - 0.065, {'Statistical difference', 'between the two videos'}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');   % a text object: MATLAB left-aligns the lines of a multi-line label
else
    qlab = {sprintf('< %.1f', Q.hi(1)), sprintf('%.1f%s%.1f', Q.lo(2), en, Q.hi(2)), sprintf('%.1f%s%.1f', Q.lo(3), en, Q.hi(3)), sprintf('> %.1f', Q.lo(4))};
    figstyle.set_ticks(ax_c, 'x', 1:4, qlab);
    figstyle.set_label(ax_c, 'x', sprintf('Difference in\\newlinepredicted metallicity'));
end
figstyle.set_ticks(ax_c, 'y', 0.45:0.1:0.75, '%.2f');
figstyle.set_label(ax_c, 'y', 'AUC, disagreement test');

%% (b) by illumination environment and by shape
Bk = readtable(fullfile(mdl, 'within_trial_breakdown.csv'), 'TextType', 'string');   % python/within_trial_breakdown.py
b_all = Bk(Bk.group == "all", :);
fprintf('  all pairs: AUC %.3f, null %.3f +- %.3f, z %.1f; per observer %.3f +- %.3f (SE), %d of 19 above 0.5\n', b_all.within_trial_auc, b_all.null_mean, b_all.null_sd, b_all.z, b_all.obs_mean, b_all.obs_se, b_all.n_obs_above);
ax_d = ax_cm(1.2, top2, 6.0, h2);   % wider: seven bars
keys = ["all", strcat("env_", string(col.envs)), strcat("geom_", string(col.geoms))];   % all pairs leftmost, then environments, then shapes
xs = [1 2.3 3.3 4.3 5.3 6.6 7.6]; cs = [figstyle.INK; col.env; col.geom];   % all pairs in black
for k = 1:numel(keys)
    b = Bk(Bk.group == keys(k), :);
    fill(ax_d, xs(k) + 0.42 * [-1 1 1 -1], b.null_mean + 2 * b.null_sd * [-1 -1 1 1], col.null, 'EdgeColor', 'none');
end
for k = 1:numel(keys)
    b = Bk(Bk.group == keys(k), :);
    bar(ax_d, xs(k), b.within_trial_auc, 0.45, 'FaceColor', cs(k, :), 'EdgeColor', cs(k, :), 'LineWidth', 0.4, 'BaseValue', 0.5);
    errorbar(ax_d, xs(k), b.within_trial_auc, b.obs_se, 'LineStyle', 'none', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_ERR, 'CapSize', 0);
    fprintf('  %-12s (%d pairs): AUC %.3f  null %.3f +- %.3f  z %.1f  per observer %.3f +- %.3f\n', keys(k), b.n_pairs, b.within_trial_auc, b.null_mean, b.null_sd, b.z, b.obs_mean, b.obs_se);
end
figstyle.ref_line(ax_d, 'h', 0.5, '');
xlim(ax_d, [0.4 8.2]); ylim(ax_d, [0.45 0.75]);
figstyle.set_ticks(ax_d, 'x', xs, [{'All'}, col.envs, col.geoms]);
ax_d.XTickLabelRotation = 0;
figstyle.set_ticks(ax_d, 'y', 0.45:0.1:0.75, '%.2f');
figstyle.set_label(ax_d, 'y', 'AUC, disagreement test');
text(ax_d, 3.8, 0.45 - 0.065, 'Environment', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');
text(ax_d, 7.1, 0.45 - 0.065, 'Shape', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');

%% (d) time course: the AUC for the time windows, against the permutation null
ax_e = ax_cm(13.7, top2, 3.9, h2);
fill(ax_e, [0.3 4.7 4.7 0.3], mean(nm) + 2 * mean(ns) * [-1 -1 1 1], col.null, 'EdgeColor', 'none');
se_b = cellfun(@(c) std(Wb.(c)) / sqrt(height(Wb)), {'first2s', 'early_half', 'late_half', 'stats'});
for k = 1:4
    bar(ax_e, k, wt(k), 0.55, 'FaceColor', cols(k, :), 'EdgeColor', col.accent, 'LineWidth', 0.4, 'BaseValue', 0.5);
    errorbar(ax_e, k, wt(k), se_b(k), 'LineStyle', 'none', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_ERR, 'CapSize', 0);
end
figstyle.ref_line(ax_e, 'h', 0.5, '');
xlim(ax_e, [0.3 4.7]); ylim(ax_e, [0.45 0.75]);
figstyle.set_ticks(ax_e, 'x', 1:4, {'', '', '', ''}); center_xticklabels(ax_e, {{'First', '2 s'}, {'First', 'half'}, {'Second', 'half'}, {'Whole', 'trial'}});
figstyle.set_ticks(ax_e, 'y', 0.45:0.1:0.75, '%.2f');
figstyle.set_label(ax_e, 'y', 'AUC, disagreement test');
text(ax_e, 2.5, 0.45 - 0.065, 'Part of the video', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');
for k = 1:4
    fprintf('  %-22s within-trial AUC %.3f  null %.3f +- %.3f  z %.1f\n', rows.feature_set(k), rows.within_trial_auc(k), ...
        rows.within_trial_null_mean(k), rows.within_trial_null_sd(k), rows.within_trial_z(k));
end

figstyle.panel_labels(fig, [ax_d ax_c ax_e], {'(b)', '(c)', '(d)'});
ov = figstyle.overlay(fig);
text(ov, 0.15 / W, 1 - 0.55 / H, '(a)', 'Units', 'normalized', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_PANEL, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
figstyle.save(fig, fullfile(project_root, 'figs', 'fig7'));
close(fig);
