% fig2_model
% Figure 2: how the own-video model is built and tested, on real data.
%
% (a) one video becomes 116 statistics (z-scored within the observer);
% (b) the responses of all observers to all trials; the observer being
%     predicted is left out and a logistic regression s = w.x + b is fitted
%     to the other 18 observers' videos and responses;
% (c) the fitted model scores the left-out observer's 120 videos; the score
%     is the log-odds of a "metal" response (sigmoid), and the AUC against
%     the observer's own responses is the overall test (Figure 4);
% (d) the disagreement test: the scores of the 19 videos of one trial
%     (from the model fitted without that stimulus), colored by the
%     response; among the pairs of observers who disagreed, how often did
%     the "metal" observer's video score higher? Over all trials this is the
%     AUC of the disagreement test; the null shuffles the responses among
%     the observers of each trial.
% Output: figs/fig2.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
mdl = fullfile(project_root, 'results', 'models');
ex_dir = fullfile(project_root, 'data', 'example_frames');

T = readtable(fullfile(project_root, 'data', 'trials.csv'), 'TextType', 'string');
S = readtable(fullfile(mdl, 'object_relational_flow_C0.1_scores.csv'), 'TextType', 'string');
X = readtable(fullfile(ex_dir, 'model_illustration.csv'), 'TextType', 'string');
Oa = readtable(fullfile(mdl, 'object_relational_flow_C0.1_observer_auc.csv'));
M = readtable(fullfile(project_root, 'results', 'summary', 'model_summary.csv'), 'TextType', 'string');
m = M(M.feature_set == "object_relational_flow" & abs(M.C - 0.1) < 1e-9, :);
meta = @(k) X.w(X.feature == k);
obs_ex = meta("observer"); trial = meta("trial");
tr_ex = T(double(string(T.participant)) == obs_ex & T.trial == trial, :);
c_metal = col.accent; c_glass = 0.15 * col.accent + 0.85 * [1 1 1];

W = figstyle.WIDTH_2COL; H = 8.3;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);
img_cm = @(l, t, w, h) axes('Parent', fig, 'Units', 'normalized', 'Position', [l / W, (H - t - h) / H, w / W, h / H]);
ov = figstyle.overlay(fig);
note = @(x_cm, y_cm, s, varargin) text(ov, x_cm / W, 1 - y_cm / H, s, 'Units', 'normalized', 'FontName', figstyle.FONT, ...
    'FontSize', figstyle.FS_LEGEND, 'Color', figstyle.INK, 'VerticalAlignment', 'top', varargin{:});
arrow = @(x1, y1, x2, y2) annotation(fig, 'arrow', [x1 x2] / W, 1 - [y1 y2] / H, 'Color', figstyle.GRAY_TEXT, ...
    'LineWidth', figstyle.LW_THIN, 'HeadLength', 4, 'HeadWidth', 4);

%% (a) one video -> 116 statistics
top_a = 1.0; fw = 1.15; gap = 0.08; left_a = 0.5;
fr = dir(fullfile(ex_dir, 'timecourse_ctx_*s.png'));
tok = cellfun(@(n) regexp(n, '^timecourse_ctx_(\d+)_[0-9.]+s\.png$', 'tokens', 'once'), {fr.name}, 'UniformOutput', false);
keep = ~cellfun(@isempty, tok); fr = fr(keep); idx = cellfun(@(t) str2double(t{1}), tok(keep));
[~, ord] = sort(idx); fr = fr(ord([1 5 8]));
for k = 1:3
    ax = img_cm(left_a + (k - 1) * (fw + gap), top_a + 0.35, fw, fw);
    image(ax, imread(fullfile(ex_dir, fr(k).name))); axis(ax, 'image', 'off');
    t = sscanf(fr(k).name, 'timecourse_ctx_%*d_%fs.png');
    text(ax, 112, -6, sprintf('%.1f s', t), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND);
end
note(left_a, top_a - 0.35, sprintf('One video (observer %d, trial %d)', obs_ex, trial), 'Color', figstyle.GRAY_TEXT);
x_arrow = left_a + 3 * (fw + gap) + 0.05;
arrow(x_arrow, top_a + 0.35 + fw / 2, x_arrow + 0.55, top_a + 0.35 + fw / 2);
ax_a = ax_cm(x_arrow + 1.5, top_a + 0.3, 3.3, 1.4);
xz = X.x(~isnan(X.x));
bar(ax_a, 1:numel(xz), xz, 0.8, 'FaceColor', c_metal, 'EdgeColor', 'none');
xlim(ax_a, [0 numel(xz) + 1]); ylim(ax_a, [-3 3]);
figstyle.set_ticks(ax_a, 'x', [1 50 100 116], '%d'); figstyle.set_ticks(ax_a, 'y', -2:2:2, '%d');
figstyle.set_label(ax_a, 'x', 'Statistic'); figstyle.set_label(ax_a, 'y', 'z-score');
title(ax_a, sprintf('x: %d statistics of this video', numel(xz)), 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, 'FontWeight', 'normal');

%% (b) all responses; one observer left out; fit to the others
pids = unique(T.participant); trials = unique(T.trial);
R = nan(numel(pids), numel(trials));
for i = 1:numel(pids)
    r = T(T.participant == pids(i), :);
    [~, j] = ismember(r.trial, trials); R(i, j) = r.resp;
end
left_b = 10.4; top_b = 1.0; wb = 4.6; hb = 1.7;
ax_b = ax_cm(left_b, top_b, wb, hb);
imagesc(ax_b, 1:numel(trials), 1:numel(pids), R); colormap(ax_b, [c_glass; c_metal]); caxis(ax_b, [0 1]);
set(ax_b, 'YDir', 'reverse', 'XGrid', 'off', 'YGrid', 'off', 'XMinorGrid', 'off', 'YMinorGrid', 'off');
xlim(ax_b, [0.5 numel(trials) + 0.5]); ylim(ax_b, [0.5 numel(pids) + 0.5]);
figstyle.set_ticks(ax_b, 'x', [1 40 80 120], '%d'); figstyle.set_ticks(ax_b, 'y', [1 10 19], '%d');
figstyle.set_label(ax_b, 'x', 'Trial (120)'); figstyle.set_label(ax_b, 'y', 'Observer (19)');
i_left = find(double(string(pids)) == obs_ex);
rectangle(ax_b, 'Position', [0.5 i_left - 0.5 numel(trials) 1], 'EdgeColor', figstyle.INK, 'LineWidth', 0.8);
text(ax_b, numel(trials) + 2, i_left, sprintf('observer %d: left out', obs_ex), 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'middle');
text(ax_b, numel(trials) + 2, 1 + (numel(pids) - 1) * 0.55, {'the other 18:', 'fit s = w\cdotx + b'}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'middle');
hk = [patch(ax_b, nan, nan, c_metal, 'EdgeColor', 'none'), patch(ax_b, nan, nan, c_glass, 'EdgeColor', 'none')];
lg = legend(ax_b, hk, {'said "metal"', 'said "glass"'}, 'Box', 'off', 'FontSize', figstyle.FS_LEGEND, 'Orientation', 'horizontal', 'Location', 'none', 'Units', 'normalized');
lg.Position(1:2) = [ax_b.Position(1), ax_b.Position(2) + ax_b.Position(4) + 0.015];

%% (c) the score and the sigmoid, for the left-out observer: three of their videos
s3 = S(double(string(S.participant)) == obs_ex, :);
E = readtable(fullfile(ex_dir, 'score_examples.csv'), 'TextType', 'string');
left_c = 1.2; top_c = 3.9; wc = 7.0; hc = 3.4;
ax_c = ax_cm(left_c, top_c, wc, hc);
xl = [-7 7]; yl = [-0.12 1.12];
d2cm_x = @(x) left_c + (x - xl(1)) / diff(xl) * wc;      % data -> cm from the left
d2cm_y = @(y) top_c + (yl(2) - y) / diff(yl) * hc;       % data -> cm from the top
ss = linspace(xl(1), xl(2), 200); sig = @(s) 1 ./ (1 + exp(-s));
plot(ax_c, ss, sig(ss), '-', 'Color', figstyle.GRAY_TEXT, 'LineWidth', figstyle.LW_LINE);
sx = s3.score(s3.trial == trial);
ex = [E.score(E.role == "low"), sx, E.score(E.role == "high")]; ex_resp = [E.resp(E.role == "low"), 0, E.resp(E.role == "high")];
for k = 1:3
    c = c_glass; if ex_resp(k) == 1, c = c_metal; end
    plot(ax_c, ex(k), sig(ex(k)), 'o', 'MarkerSize', figstyle.MS + 1, 'MarkerFaceColor', c, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
end
% frames of the lowest- and highest-scoring videos: low at the top left, high at the bottom right
fw2 = 0.85; g2 = 0.06; sw = 3 * fw2 + 2 * g2;
strip = @(stem, l, t) local_strip(fig, ex_dir, stem, l, t, fw2, g2, W, H);
l_lo = left_c + 0.25; t_lo = top_c + 0.12; strip("score_low", l_lo, t_lo);
l_hi = left_c + wc - sw - 0.1; t_hi = top_c + hc - fw2 - 0.12; strip("score_high", l_hi, t_hi);
names = {'glass', 'metal'};
lab = @(role) sprintf('s = %+.1f, said "%s"', E.score(E.role == role), names{E.resp(E.role == role) + 1});
y_lo = (yl(2) - (t_lo + fw2 + 0.08 - top_c) / hc * diff(yl));              % data y just below the low strip
x_lo = (l_lo + sw + 0.15 - left_c) / wc * diff(xl) + xl(1);                 % to the right of the strip
text(ax_c, x_lo, (yl(2) - (t_lo + fw2 / 2 - 0.12 - top_c) / hc * diff(yl)), strsplit(lab("low"), ', '), 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'middle');
plot(ax_c, [ex(1) ex(1)], [y_lo, sig(ex(1)) + 0.07], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
y_hi = (yl(2) - (t_hi - 0.42 - top_c) / hc * diff(yl));                    % data y above the high strip and its times
x_hi = (l_hi + sw / 2 - left_c) / wc * diff(xl) + xl(1);
text(ax_c, x_hi, y_hi, lab("high"), 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
plot(ax_c, [ex(3) ex(3)], [y_hi + 0.13, sig(ex(3)) - 0.07], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
% the video of (a)
text(ax_c, sx - 0.5, 0.47, {'the video', 'of (a):', sprintf('s = %+.1f', sx)}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
plot(ax_c, [sx - 0.4, sx - 0.05], [0.40, sig(sx) + 0.05], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
xlim(ax_c, xl); ylim(ax_c, yl);
figstyle.set_ticks(ax_c, 'x', -6:3:6, '%d'); figstyle.set_ticks(ax_c, 'y', 0:0.5:1, '%.1f');
figstyle.set_label(ax_c, 'x', 'Predicted metallicity s = w\cdotx + b'); figstyle.set_label(ax_c, 'y', 'P(metal) = 1 / (1 + e^{-s})');
auc3 = Oa.auc_own(Oa.participant == obs_ex);
note(left_c + wc / 2, top_c - 0.42, sprintf('observer %d: AUC %.2f over the 120 videos', obs_ex, auc3), 'Color', figstyle.GRAY_TEXT, 'HorizontalAlignment', 'center');
fprintf('  observer %d: AUC %.3f, example score %+.2f; low %+.2f (trial %d), high %+.2f (trial %d)\n', obs_ex, auc3, sx, ex(1), E.trial(E.role == "low"), ex(3), E.trial(E.role == "high"));

%% (d) the within-trial test on one trial: the 19 videos, two of them shown
st = S(S.trial == trial, :); st.score = st.score_wt; st = sortrows(st, 'score');   % the disagreement test's scores (leave-one-stimulus-out)
Et = readtable(fullfile(ex_dir, 'trial_examples.csv'), 'TextType', 'string');
left_d = 9.8; top_d = 4.4; wd = 7.0; hd = 0.9;
ax_d = ax_cm(left_d, top_d, wd, hd);
xld = [-4 4];
rng(2); yj = 0.2 + 0.6 * rand(height(st), 1);
for r = [0 1]
    sel = st.resp == r; c = c_glass; if r == 1, c = c_metal; end
    plot(ax_d, st.score(sel), yj(sel), 'o', 'MarkerSize', figstyle.MS - 0.5, 'MarkerFaceColor', c, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.3);
end
k = find(double(string(st.participant)) == obs_ex);
plot(ax_d, st.score(k), yj(k), 'o', 'MarkerSize', figstyle.MS + 2, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
plot(ax_d, [st.score(k) st.score(k)], [yj(k) + 0.1, 1.12], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN, 'Clipping', 'off');
text(ax_d, st.score(k), 1.2, sprintf('observer %d, the video of (a)', obs_ex), 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
xlim(ax_d, xld); ylim(ax_d, [0 1]);
figstyle.set_ticks(ax_d, 'x', -4:2:4, '%d'); ax_d.YTick = []; ax_d.YGrid = 'off'; ax_d.YMinorGrid = 'off';
figstyle.set_label(ax_d, 'x', sprintf('Predicted metallicity s, trial %d', trial));
% the lowest- and highest-scoring videos of the trial, three frames each, under the axis
t_s = top_d + hd + 1.05;
d2cm_xd = @(x) left_d + (x - xld(1)) / diff(xld) * wd;
for role = ["low", "high"]
    r = Et(Et.role == role, :); kk = find(double(string(st.participant)) == double(string(r.participant)));
    if role == "low", l_s = left_d + 0.1; else, l_s = left_d + wd - sw - 0.1; end
    strip("trial_" + role, l_s, t_s);
    sk = st.score(kk);   % the score plotted above, not the copy in trial_examples.csv
    plot(ax_d, sk * [1 1], [yj(kk) - 0.1, -(t_s - 0.32 - top_d - hd) / hd], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN, 'Clipping', 'off');
    plot(ax_d, sk, yj(kk), 'o', 'MarkerSize', figstyle.MS + 2, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
    note(l_s + sw / 2, t_s + fw2 + 0.1, sprintf('observer %d: s = %+.1f, said "%s"', double(string(r.participant)), sk, names{r.resp + 1}), 'HorizontalAlignment', 'center');
end
n_m = sum(st.resp == 1); n_g = sum(st.resp == 0);
ok = 0; for i = find(st.resp == 1)', ok = ok + sum(st.score(i) > st.score(st.resp == 0)); end
fprintf('  trial %d: %d metal, %d glass, %d pairs, metal higher in %d\n', trial, n_m, n_g, n_m * n_g, ok);

figstyle.panel_labels(fig, [ax_b ax_c ax_d], {'(b)', '(c)', '(d)'});
text(ov, 0.15 / W, 1 - 0.45 / H, '(a)', 'Units', 'normalized', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_PANEL, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
figstyle.save(fig, fullfile(project_root, 'figs', 'fig2'));
close(fig);

function local_strip(fig, ex_dir, stem, l, t, fw, g, W, H)
% three object-plus-surround frames of one video (<stem>_ctx_<k>_<t>s.png), with their times
fr = dir(fullfile(ex_dir, sprintf('%s_ctx_*s.png', stem)));
tok = cellfun(@(n) regexp(n, '_ctx_(\d+)_([0-9.]+)s\.png$', 'tokens', 'once'), {fr.name}, 'UniformOutput', false);
idx = cellfun(@(x) str2double(x{1}), tok); ts = cellfun(@(x) str2double(x{2}), tok);
[~, ord] = sort(idx); fr = fr(ord); ts = ts(ord);
for k = 1:numel(fr)
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [(l + (k - 1) * (fw + g)) / W, (H - t - fw) / H, fw / W, fw / H]);
    image(ax, imread(fullfile(ex_dir, fr(k).name))); axis(ax, 'image', 'off');
    text(ax, 112, -6, sprintf('%.1f s', ts(k)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND - 1);
end
end
