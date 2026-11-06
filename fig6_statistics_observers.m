% fig6_statistics_observers
% Figure 6: which image statistics predict each observer's responses (the 116 statistics of the main model).
%
% For each observer, the AUC of every statistic over that observer's 120
% trials (> 0.5: a higher value went with "metal", < 0.5 with "glass"), by
% family (the 16 statistics of Figure 5 filled magenta); the best statistic in each direction is named above (metal) and
% below (glass) the column; the dash-dotted line is the own-video model,
% which uses all the statistics (leave-one-observer-out).
% Output: figs/fig6.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
mdl = fullfile(project_root, 'results', 'models');

G = readtable(fullfile(mdl, 'single_statistics.csv'), 'TextType', 'string');
B = readtable(fullfile(mdl, 'single_statistics_by_observer.csv'), 'TextType', 'string');
Oa = readtable(fullfile(mdl, 'object_relational_flow_C0.1_observer_auc.csv'), 'TextType', 'string');
B.participant = string(B.participant); Oa.participant = string(Oa.participant);

W = figstyle.WIDTH_2COL; H = 13.7;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);

%% every statistic for every observer: raw AUC (> 0.5 metal direction, < 0.5 glass direction)
feats = G.feature(ismember(G.feature, string(B.Properties.VariableNames)));
nf = numel(feats);
Gm = sortrows(G, 'auc', 'descend'); Gg = sortrows(G, 'auc', 'ascend');
top16 = ismember(feats, [Gm.feature(1:8); Gg.feature(1:8)]);   % the statistics of Figure 5: the 8 most "metal" and the 8 most "glass"
pids = sortrows(Oa, 'auc_own', 'descend').participant;   % observers ordered by how well the full model predicts them
no = numel(pids);
[~, bi] = ismember(pids, B.participant); [~, oi] = ismember(pids, Oa.participant);
Mr = nan(nf, no);
for i = 1:nf, Mr(i, :) = B.(char(feats(i)))(bi)'; end
fam = zeros(nf, 1);
for i = 1:nf
    f = char(feats(i));
    if startsWith(f, 'coverage') || startsWith(f, 'div_') || startsWith(f, 'sfm_') || startsWith(f, 'flow_'), fam(i) = 5;
    elseif startsWith(f, 'obj_bg_'), fam(i) = 4;
    elseif startsWith(f, 'bg_'), fam(i) = 3;
    elseif startsWith(f, 'chroma') || startsWith(f, 'area') || startsWith(f, 'diff_energy'), fam(i) = 2;
    else, fam(i) = 1; end
end
fam_names = {'Object luminance', 'Object color, size, change', 'Surround', 'Object vs surround', 'Motion (optic flow)'};
fam_col = [col.ramp(5, :); col.ramp(3, :); 0.62 0.62 0.62; col.ramp(1, :); 0.30 0.30 0.30];
top_b = 2.6; hb = 6.6;
wb = 15.0; ax_b = ax_cm(1.3, top_b, wb, hb);   % room at the right for the direction notes and thumbnails
yl_b = [0 1];
rng(3);
[~, im] = max(Mr, [], 1); [~, ig] = min(Mr, [], 1);
lab_fs = figstyle.FS_LEGEND - 0.5;
for j = 1:no
    xj = j + 0.56 * (rand(nf, 1) - 0.5);
    for k = 1:5
        sel = fam == k & ~top16;
        scatter(ax_b, xj(sel), Mr(sel, j), 5, 'o', 'MarkerFaceColor', fam_col(k, :), 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);
    end
    scatter(ax_b, xj(top16), Mr(top16, j), 9, 'o', 'MarkerFaceColor', col.highlight, 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.9);   % the 16 statistics of Figure 5
    plot(ax_b, j + [-0.33 0.33], Oa.auc_own(oi(j)) * [1 1], '-.', 'Color', figstyle.GRAY_TEXT, 'LineWidth', 0.8);   % the own-video model, all statistics
    % best statistic in the metal direction: named above the column
    plot(ax_b, j, Mr(im(j), j), 'o', 'MarkerSize', figstyle.MS + 1, 'MarkerFaceColor', col.metal_dir, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
    nm = local_lines(local_short(stat_name(feats(im(j)))));
    y_lab = yl_b(2) + (0.04 + 0.11 * mod(j, 3)) * diff(yl_b);
    text(ax_b, j, y_lab, nm, 'FontName', figstyle.FONT, 'FontSize', lab_fs, 'Color', col.metal_dir, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Clipping', 'off');
    plot(ax_b, [j j], [Mr(im(j), j) + 0.03, y_lab - 0.01], '-', 'Color', [0.65 0.65 0.65], 'LineWidth', figstyle.LW_THIN, 'Clipping', 'off');
    % best statistic in the glass direction: named below the column
    plot(ax_b, j, Mr(ig(j), j), 'o', 'MarkerSize', figstyle.MS + 1, 'MarkerFaceColor', col.glass_dir, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
    nm = local_lines(local_short(stat_name(feats(ig(j)))));
    y_lab = yl_b(1) - (0.06 + 0.11 * mod(j, 3)) * diff(yl_b);
    text(ax_b, j, y_lab, nm, 'FontName', figstyle.FONT, 'FontSize', lab_fs, 'Color', col.ramp(3, :), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');
    plot(ax_b, [j j], [Mr(ig(j), j) - 0.03, y_lab + 0.01], '-', 'Color', [0.65 0.65 0.65], 'LineWidth', figstyle.LW_THIN, 'Clipping', 'off');
end
plot(ax_b, [0.3 no + 0.7], [0.5 0.5], '--', 'Color', figstyle.GRAY_TEXT, 'LineWidth', figstyle.LW_AXIS);   % chance
xlim(ax_b, [0.3 no + 0.7]); ylim(ax_b, yl_b);
figstyle.set_ticks(ax_b, 'x', 1:no, cellstr(string(double(pids))));
ax_b.XAxisLocation = 'top'; ax_b.XTickLabel = repmat({''}, 1, no);
ax_b.XAxis.Visible = 'off'; ax_b.Box = 'off';      % no lines at the top (AUC 1) or bottom (AUC 0) of the panel
figstyle.set_ticks(ax_b, 'y', 0:0.25:1, '%.2f');
figstyle.set_label(ax_b, 'y', 'AUC for the observer''s responses');
% which direction means what, outside the panel at the right
% which direction means what, outside the panel at the right
x_note = no + 0.7 + 0.75 / (wb / (no + 0.4));            % 0.75 cm right of the panel, in data units; notes, arrow and thumbnails centred on it
% a metal and a glass object above and below the notes (Sky, bumpy, physical metallicity 1 and 0; the model's patches)
ex_dir = fullfile(project_root, 'data', 'example_frames'); tw = 1.0; x_th = 1.3 + wb + 0.75 - tw / 2;
axi = axes('Parent', fig, 'Units', 'normalized', 'Position', [x_th / W, (H - top_b + 0.08) / H, tw / W, tw / H]);
image(axi, imread(fullfile(ex_dir, 'ctx_Sky_Bumpy_m1.png'))); axis(axi, 'image', 'off');
axi = axes('Parent', fig, 'Units', 'normalized', 'Position', [x_th / W, (H - top_b - hb - 0.08 - tw) / H, tw / W, tw / H]);
image(axi, imread(fullfile(ex_dir, 'ctx_Sky_Bumpy_m0.png'))); axis(axi, 'image', 'off');
text(ax_b, x_note, 0.99, {'More', 'metal'}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', col.metal_dir, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');
text(ax_b, x_note, 0.01, {'More', 'glass'}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, 'Color', col.ramp(3, :), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Clipping', 'off');
x_arr = ax_b.Position(1) + ax_b.Position(3) + 0.75 / W;      % the same point in figure units
annotation(fig, 'doublearrow', [x_arr x_arr], ax_b.Position(2) + [0.12 0.88] * ax_b.Position(4), 'Color', figstyle.GRAY_TEXT, 'LineWidth', figstyle.LW_THIN, 'Head1Length', 4, 'Head1Width', 4, 'Head2Length', 4, 'Head2Width', 4);
hk = gobjects(1, 9);
for k = 1:5, hk(k) = plot(ax_b, nan, nan, 'o', 'MarkerFaceColor', fam_col(k, :), 'MarkerEdgeColor', 'none', 'MarkerSize', 4); end
hk(6) = plot(ax_b, nan, nan, 'o', 'MarkerFaceColor', col.highlight, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
hk(7) = plot(ax_b, nan, nan, 'o', 'MarkerSize', figstyle.MS + 1, 'MarkerFaceColor', col.metal_dir, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
hk(8) = plot(ax_b, nan, nan, 'o', 'MarkerSize', figstyle.MS + 1, 'MarkerFaceColor', col.glass_dir, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.8);
hk(9) = plot(ax_b, nan, nan, '-.', 'Color', figstyle.GRAY_TEXT, 'LineWidth', 0.8);
lg = legend(ax_b, hk, [fam_names, {'the 16 statistics of Figure 5', 'best for metal (named above; higher value, metal response)', 'best for glass (named below; higher value, glass response)', 'own-video model, all statistics'}], ...
    'Box', 'off', 'FontSize', figstyle.FS_LEGEND, 'Orientation', 'horizontal', 'Location', 'none', 'Units', 'normalized', 'NumColumns', 3, 'AutoUpdate', 'off');
lg.ItemTokenSize = [14 18];                           % shorter line and marker samples in the legend
lg.Position(1:2) = [ax_b.Position(1) - 0.01, 0.25 / H];
fprintf('  %d statistics; best statistic per observer (metal direction | glass direction):\n', nf);
for j = 1:no, fprintf('      observer %2d: %-28s %.3f | %-28s %.3f  (model %.3f)\n', double(pids(j)), feats(im(j)), Mr(im(j), j), feats(ig(j)), Mr(ig(j), j), Oa.auc_own(oi(j))); end
fprintf('  distinct best (metal) %d, (glass) %d\n', numel(unique(im)), numel(unique(ig)));

figstyle.save(fig, fullfile(project_root, 'figs', 'fig6'));
close(fig);

function nm = local_lines(s)
% label lines: 'name, summary' -> {'name', '(summary)'}; the object-versus-surround names break after 'obj./surr.'
parts = strsplit(s, ', ');
if numel(parts) > 1, parts{end} = ['(' parts{end} ')']; end      % the summary over the trial in parentheses
if startsWith(parts{1}, 'obj./surr. ')
    nm = [{'obj./surr.', parts{1}(12:end)}, parts(2:end)];
elseif strcmp(parts{1}, 'maximum luminance')                    % a long name: two lines
    nm = [{'maximum', 'luminance'}, parts(2:end)];
else
    nm = parts;
end
end

function s = local_short(s)
% shorter statistic names for the labels above the columns
s = strrep(s, 'Object/surround', 'obj./surr.'); s = strrep(s, 'Surround maximum luminance', 'surr. max lum.');
s = strrep(s, 'Surround luminance SD', 'surr. lum. SD'); s = strrep(s, 'Surround luminance', 'surr. luminance');
s = strrep(s, 'Surround RMS contrast', 'surr. RMS contrast'); s = strrep(s, 'Surround chroma', 'surr. chroma');
s = strrep(s, 'Highlight coverage', 'highlight cov.'); s = strrep(s, 'Luminance kurtosis', 'lum. kurtosis'); s = strrep(s, 'Luminance skewness', 'lum. skewness');
s = strrep(s, 'late minus early', ['late ' char(8722) ' early']);
if numel(s) > 1 && ~isstrprop(s(2), 'upper'), s = [lower(s(1)) s(2:end)]; end
end
