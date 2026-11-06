% fig5_statistics
% Figure 5: which image statistics predict the responses (group).
%
% For the 8 statistics that predict "metal" best (highest mean AUC) and the 8
% that predict "glass" best (lowest), the AUC of the statistic (z-scored within
% observer) against the observer's responses, mean over the 19 observers with
% its standard error (> 0.5: a higher value went with a "metal" response,
% < 0.5 with a "glass" response); the gray band is +-2 SD of the permutation null.
% Statistics along x, AUC along y: the same orientation as Figure 6.
% Output: figs/fig5.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
mdl = fullfile(project_root, 'results', 'models');

G = readtable(fullfile(mdl, 'single_statistics.csv'), 'TextType', 'string');
n_obs = 19;

W = figstyle.WIDTH_2COL; H = 8.9;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);

%% the 8 statistics that predict "metal" best (left) and the 8 that predict "glass" best (right)
Gm = sortrows(G, 'auc', 'descend'); Gg = sortrows(G, 'auc', 'ascend');
Gs = [Gm(1:8, :); flipud(Gg(1:8, :))];       % most "metal" at the left, most "glass" at the right
n = height(Gs);
ph = 5.2; ax_a = ax_cm(1.5, 0.9, W - 2.0, ph);
fill(ax_a, [0.3 n + 0.7 n + 0.7 0.3], 0.5 + 2 * mean(Gs.null_sd) * [-1 -1 1 1], col.null, 'EdgeColor', 'none');
for i = 1:n
    v = Gs.auc(i); se = Gs.auc_sd(i) / sqrt(n_obs);
    c = col.metal_dir; if v < 0.5, c = col.glass_dir; end
    bar(ax_a, i, v, 0.65, 'BaseValue', 0.5, 'FaceColor', c, 'EdgeColor', 'none');
    plot(ax_a, [i i], v + se * [-1 1], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
    fprintf('  %-32s AUC %.3f (SD %.3f, SE %.3f over observers; %d/19 in the group direction)\n', Gs.feature(i), v, Gs.auc_sd(i), se, Gs.n_same_direction(i));
end
plot(ax_a, [0.3 n + 0.7], [0.5 0.5], '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_AXIS);
lab = arrayfun(@(k) local_lines(local_short(stat_name(Gs.feature(k)))), 1:n, 'UniformOutput', false);
figstyle.set_ticks(ax_a, 'x', 1:n, repmat({''}, 1, n));
figstyle.set_ticks(ax_a, 'y', 0.3:0.1:0.7, '%.1f');
xlim(ax_a, [0.3 n + 0.7]); ylim(ax_a, [0.27 0.73]);
% one word per line, odd and even columns at two heights so that neighbours do not overlap
blank = repmat({''}, 1, n); lab_odd = blank; lab_even = blank;
lab_odd(1:2:n) = lab(1:2:n); lab_even(2:2:n) = lab(2:2:n);
center_xticklabels(ax_a, lab_odd, 0.12);
center_xticklabels(ax_a, lab_even, 0.12 + 1.25);
yl = ylim(ax_a);
for k = 2:2:n     % a vertical leader from the axis down to the lower label of the even columns
    plot(ax_a, [k k], yl(1) - [0, (0.12 + 1.25 - 0.08) / ph * diff(yl)], '-', 'Color', [0.7 0.7 0.7], 'LineWidth', figstyle.LW_THIN, 'Clipping', 'off');
end
figstyle.set_label(ax_a, 'y', 'AUC for the observer''s responses');
% what the two sides mean, above the panel
text(ax_a, 0.5, 0.74, 'Higher value, more metal', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', col.metal_dir, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Clipping', 'off');
text(ax_a, n + 0.5, 0.74, 'Lower value, more glass', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', col.ramp(3, :), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Clipping', 'off');
figstyle.save(fig, fullfile(project_root, 'figs', 'fig5'));
close(fig);

function nm = local_lines(s)
% label lines: 'name, summary' -> {'name', '(summary)'}; the object-versus-surround names break after 'obj./surr.'
parts = strsplit(s, ', ');
if numel(parts) > 1, parts{end} = ['(' parts{end} ')']; end      % the summary over the trial in parentheses
if startsWith(parts{1}, 'obj./surr. ')
    nm = [{'obj./surr.', parts{1}(12:end)}, parts(2:end)];
else
    nm = parts;
end
% every word on its own line (the tick labels are narrow); 'late − early' stays on one line
out = {};
for i = 1:numel(nm)
    if contains(nm{i}, char(8722)), out{end + 1} = nm{i}; else, out = [out, strsplit(nm{i}, ' ')]; end
end
nm = out;
end

function s = local_short(s)
% shorter statistic names for the tick labels (as in Figure 6)
s = strrep(s, 'Object/surround', 'obj./surr.'); s = strrep(s, 'Surround maximum luminance', 'surr. max lum.');
s = strrep(s, 'Surround luminance SD', 'surr. lum. SD'); s = strrep(s, 'Surround luminance', 'surr. luminance');
s = strrep(s, 'Surround RMS contrast', 'surr. RMS contrast'); s = strrep(s, 'Surround chroma', 'surr. chroma');
s = strrep(s, 'Highlight coverage', 'highlight cov.'); s = strrep(s, 'Luminance kurtosis', 'lum. kurtosis'); s = strrep(s, 'Luminance skewness', 'lum. skewness');
s = strrep(s, 'late minus early', ['late ' char(8722) ' early']);
if numel(s) > 1 && ~isstrprop(s(2), 'upper'), s = [lower(s(1)) s(2:end)]; end
end
