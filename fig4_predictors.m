% fig4_predictors
% Figure 4: how well each predictor accounts for the observers' responses.
%
% (a) AUC for each observer's responses (leave-one-observer-out; box plots
%     over the 19 observers) of the statistics of the own video, the same
%     statistics reduced to 20 principal components, and physical metallicity,
%     against the ceiling set by the other observers' responses (gray band);
% (b) the AUC of the own-video model against the number of principal
%     components the statistics are reduced to (mean and standard error over
%     observers; python/pca_sweep.py), the rightmost point being the model
%     with all the statistics;
% (c) as (a) by condition (head only, hand only, head + hand) and, last, the
%     condition without exploration (one image per trial: its 23 per-frame
%     quantities, fitted to the other 18 observers' trials of that condition;
%     python/static_block.py).
% Output: figs/fig4.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();

A = readtable(fullfile(project_root, 'results', 'summary', 'per_observer_auc.csv'), 'TextType', 'string');
% the same statistics reduced to k principal components (python/pca_sweep.py), all trials only
PCA_K = 20;
P = readtable(fullfile(project_root, 'results', 'models', 'pca_sweep_by_observer.csv'), 'TextType', 'string');
for i = 1:height(P)
    A(end + 1, :) = {P.participant(i), "image_stats_pca", P.(sprintf('k%d', PCA_K))(i), nan, nan, nan};
end
preds = {'image_stats', 'image_stats_pca', 'metal_parameter'};   % physical metallicity rightmost
labels = {sprintf('Statistics\\newline(own video)'), sprintf('Statistics\\newline(%d PCs)', PCA_K), sprintf('Physical\\newlinemetallicity')};
cols = [col.stats; 0.5 * col.stats + 0.5 * [1 1 1]; col.metal_param];
% reference, not a model: the noise ceiling, from the AUC of the proportion of the other 18 observers who said
% "metal" (lower bound) to that of the proportion including the observer's own response (upper bound; Nili et al., 2014)
ref = @(column) [A.(column)(A.predictor == "other_observers"), A.(column)(A.predictor == "all_observers")];

W = figstyle.WIDTH_2COL; H = 6.3;
fig = figstyle.figure(2, H);
ax_cm = @(l, t, w, h) figstyle.axes(fig, [l / W, (H - t - h) / H, w / W, h / H]);
top = 0.9; ph = 4.0;                               % top and height of every panel [cm]

%% (a) all trials
ax_a = ax_cm(1.3, top, 3.8, ph);
q_ref = local_strips(ax_a, A, preds, 'auc_all', cols, ref('auc_all'));
figstyle.set_ticks(ax_a, 'x', 1:3, labels); center_xticklabels(ax_a, labels);
figstyle.set_label(ax_a, 'y', 'AUC for the observer''s responses');

%% (b) the AUC against the number of principal components
kn = string(P.Properties.VariableNames); kn = kn(startsWith(kn, "k"));
ks = double(erase(kn, "k"));                        % numbers of components evaluated
Y = P{:, cellstr(kn)};                              % observers x numbers of components
full = A.auc_all(A.predictor == "image_stats");     % the model with all the statistics
n_all = 116;
x = [ks, n_all]; mu = [mean(Y, 1), mean(full)]; se = [std(Y, 0, 1), std(full)] / sqrt(size(Y, 1));
ax_b = ax_cm(6.2, top, 3.5, ph);
set(ax_b, 'XScale', 'log');
xl = [0.8 150];
fill(ax_b, xl([1 2 2 1]), q_ref([1 1 2 2]), [0.9 0.9 0.9], 'EdgeColor', 'none');   % the ceiling, as in (a)
plot(ax_b, x, mu, '-', 'Color', col.stats, 'LineWidth', figstyle.LW_LINE);
figstyle.plot_mean(ax_b, x(1:end - 1), mu(1:end - 1), se(1:end - 1), 0.5 * col.stats + 0.5 * [1 1 1], [], figstyle.MS - 1.5);
figstyle.plot_mean(ax_b, x(end), mu(end), se(end), col.stats, [], figstyle.MS - 1.5);      % all the statistics
xlim(ax_b, xl); ylim(ax_b, [0.5 1.05]);
set(ax_b, 'XTick', [1 10 100], 'XTickLabel', {'1', '10', '100'}, 'XMinorTick', 'on', 'XMinorGrid', 'off');   % logarithmic axis
figstyle.set_ticks(ax_b, 'y', 0.5:0.1:1, '%.1f');
figstyle.set_label(ax_b, 'y', 'AUC');
text(ax_b, sqrt(prod(xl)), 0.5 - 0.078, {'Number of principal', 'components (log scale)'}, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');   % a text object: MATLAB left-aligns the lines of a multi-line label
fprintf('  AUC by number of components: %s\n', strjoin(compose('%d: %.3f', x(:), mu(:)), ', '));

%% (c) by condition: the three exploration conditions, then the condition without exploration
St = readtable(fullfile(project_root, 'results', 'behaviour', 'static_observer_summary.csv'), 'TextType', 'string');
As = [table(St.participant, repmat("metal_parameter", height(St), 1), St.auc_metal_static, 'VariableNames', {'participant', 'predictor', 'auc_static'});
      table(St.participant, repmat("image_stats", height(St), 1), St.auc_stats_static, 'VariableNames', {'participant', 'predictor', 'auc_static'})];
mods = cellfun(@(b) ['auc_' b], col.blocks, 'UniformOutput', false);   % head only, hand only, head + hand, no exploration
left_c = 10.9; wc = 1.5; gc = 0.22;
axs = gobjects(1, 4);
for im = 1:4
    axs(im) = ax_cm(left_c + (im - 1) * (wc + gc), top, wc, ph);
    if im == 4
        local_strips(axs(im), As, preds([1 3]), mods{im}, cols([1 3], :), [St.auc_others_static, St.auc_all_static]);
    else
        local_strips(axs(im), A, preds([1 3]), mods{im}, cols([1 3], :), ref(mods{im}));
    end
    figstyle.set_ticks(axs(im), 'x', 1:2, {'S', 'M'});
    figstyle.panel_title(axs(im), strrep(col.block_labels{im}, 'No exploration', 'No expl.'), figstyle.INK);
    if im > 1; axs(im).YTickLabel = []; end
end
figstyle.set_label(axs(1), 'y', 'AUC');
for im = 1:4      % the same width for every condition: the y label must not shrink the first panel
    axs(im).PositionConstraint = 'innerposition';
    axs(im).Position = [(left_c + (im - 1) * (wc + gc)) / W, (H - top - ph) / H, wc / W, ph / H];
end
figstyle.panel_labels(fig, [ax_a ax_b axs(1)], {'(a)', '(b)', '(c)'});
figstyle.save(fig, fullfile(project_root, 'figs', 'fig4'));
close(fig);

function q = local_strips(ax, A, preds, column, cols, ref)
% Compact box plots over the 19 observers (utils/box_plot.m); the reference is the
% noise ceiling: a gray band from the median (over observers) of the lower bound
% (other 18 observers) to the median of the upper bound (the observer included).
% Returns the two bounds of the band.
xlim(ax, [0.4 numel(preds) + 0.6]); ylim(ax, [0.5 1.05]);
figstyle.set_ticks(ax, 'y', 0.5:0.1:1, '%.1f');
band = [median(ref(:, 1)), median(ref(:, 2))];
fill(ax, [0.4 numel(preds) + 0.6 numel(preds) + 0.6 0.4], [band(1) band(1) band(2) band(2)], [0.9 0.9 0.9], 'EdgeColor', 'none');
fprintf('  noise ceiling %-9s lower (other 18) median AUC %.3f, mean %.3f; upper (observer included) median %.3f, mean %.3f\n', column, band(1), mean(ref(:, 1)), band(2), mean(ref(:, 2)));
pids = unique(A.participant);
for i = 1:numel(pids)
    v = arrayfun(@(k) A.(column)(A.predictor == preds{k} & A.participant == pids(i)), 1:numel(preds));
    for k = 1:numel(preds) - 1
        plot(ax, [k + 0.2, k + 0.8], v(k:k + 1), '-', 'Color', [0.75 0.75 0.75], 'LineWidth', figstyle.LW_THIN);
    end
end
for k = 1:numel(preds)
    v = A.(column)(A.predictor == preds{k});
    box_plot(ax, k, v, cols(k, :));
    qq = quantile(v, [0.25 0.5 0.75]);
    fprintf('  %-18s %-9s median AUC %.3f  IQR %.3f-%.3f  mean %.3f (sd %.3f)\n', preds{k}, column, qq(2), qq(1), qq(3), mean(v), std(v));
end
xlim(ax, [0.4 numel(preds) + 0.6]); ylim(ax, [0.5 1.05]);
figstyle.set_ticks(ax, 'y', 0.5:0.1:1, '%.1f');
q = band;
end
