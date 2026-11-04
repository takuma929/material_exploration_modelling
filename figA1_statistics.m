% figA1_statistics
% Figure A1 (Appendix): how each image statistic is computed, illustrated on
% one frame of one trial (observer 5, trial 123: Sky, bumpy, metallic 0.5,
% head + hand) and on the time series of that trial. One icon per kind of
% statistic of section 2.4: on the object crop, the luminance distribution,
% the highlights, the dark pixels, the edge energy, the chroma, the object's
% area and the frame difference; on the patch, the surround and the object
% against its surround; and over the trial, the summaries of a time series,
% its lag-one autocorrelation and the time windows; and the optic-flow cues of
% Doerschner et al. (2011) on one frame pair of the same trial.
% Output: figs/figA1.{png,pdf}

clearvars; close all;
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'utils'));
col = project_colors();
ex_dir = fullfile(project_root, 'data', 'example_frames');

%% the example frame and the time series
rgb = im2double(imread(fullfile(ex_dir, 'example_object.png')));          % the object crop
% edge energy as in the pipeline: forward differences on the original crop,
% every pixel of the stencil at least 2 px inside the silhouette
lin0 = rgb; lo = rgb <= 0.04045; lin0(lo) = rgb(lo) / 12.92; lin0(~lo) = ((rgb(~lo) + 0.055) / 1.055) .^ 2.4;
Y0 = 0.2126 * lin0(:, :, 1) + 0.7152 * lin0(:, :, 2) + 0.0722 * lin0(:, :, 3);
m0 = imerode(Y0 > 0.01, strel('square', 5));
g0 = zeros(size(Y0)); gx0 = diff(Y0, 1, 2); gy0 = diff(Y0, 1, 1);
g0(1:end-1, 1:end-1) = sqrt(gx0(1:end-1, :) .^ 2 + gy0(:, 1:end-1) .^ 2);
g0(~(m0 & circshift(m0, [0 -1]) & circshift(m0, [-1 0]))) = 0;
[rgb, zoom_scale, rr, cc] = local_zoom(rgb);                                  % shown enlarged around the object
gmag = imresize(g0(rr, cc), [224 224], 'nearest');
lin = rgb; lo = rgb <= 0.04045; lin(lo) = rgb(lo) / 12.92; lin(~lo) = ((rgb(~lo) + 0.055) / 1.055) .^ 2.4;
Y = 0.2126 * lin(:, :, 1) + 0.7152 * lin(:, :, 2) + 0.0722 * lin(:, :, 3);
obj = Y > 0.01;                                                               % the object's pixels (black background)
lab = rgb2lab(rgb); chroma = sqrt(lab(:, :, 2) .^ 2 + lab(:, :, 3) .^ 2);
patch_rgb = im2double(imread(fullfile(ex_dir, 'example_patch.png')));         % the object-plus-surround patch
pmask = imread(fullfile(ex_dir, 'example_patch_mask.png')) > 127;
plin = patch_rgb; lo = patch_rgb <= 0.04045; plin(lo) = patch_rgb(lo) / 12.92; plin(~lo) = ((patch_rgb(~lo) + 0.055) / 1.055) .^ 2.4;
PY = 0.2126 * plin(:, :, 1) + 0.7152 * plin(:, :, 2) + 0.0722 * plin(:, :, 3);
c0 = local_zoom(im2double(imread(fullfile(ex_dir, 'consecutive_0.png')))); c1 = local_zoom(im2double(imread(fullfile(ex_dir, 'consecutive_1.png'))));
Yc0 = 0.2126 * c0(:, :, 1) + 0.7152 * c0(:, :, 2) + 0.0722 * c0(:, :, 3);
Yc1 = 0.2126 * c1(:, :, 1) + 0.7152 * c1(:, :, 2) + 0.0722 * c1(:, :, 3);
S = readtable(fullfile(ex_dir, 'timecourse_stats.csv'));
t = S.t_s; v = S.rms_contrast;

%% layout [cm]
W = figstyle.WIDTH_2COL; ncol = 4; cw = W / ncol;
head_h = 1.6; icon_h = 2.55; gap = 0.5; pitch = head_h + icon_h + gap; top0 = 0.15; x0 = 0.45;
H = top0 + 4 * pitch;
fig = figstyle.figure(2, H);
ov = figstyle.overlay(fig);
cell_left = @(k) x0 + mod(k - 1, ncol) * cw;
cell_top = @(k) top0 + floor((k - 1) / ncol) * pitch;
ax_cm = @(l, tp, w, h) figstyle.axes(fig, [l / W, (H - tp - h) / H, w / W, h / H]);
img_cm = @(l, tp, w, h) axes('Parent', fig, 'Units', 'normalized', 'Position', [l / W, (H - tp - h) / H, w / W, h / H]);
plot_box = @(k) deal(cell_left(k) + 0.95, cell_top(k) + head_h, cw - 1.45, icon_h - 0.5);
img_box = @(k) deal(cell_left(k) + 0.1, cell_top(k) + head_h, icon_h, icon_h);
small = {'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND};
accent = col.accent; light = col.ramp(1, :);

function heading(ov, W, H, x, y, tag, names, what, accent)
    txt = @(dy, s, varargin) text(ov, x / W, 1 - (y + dy) / H, s, 'Units', 'normalized', 'FontName', figstyle.FONT, ...
        'FontSize', figstyle.FS_LEGEND, 'VerticalAlignment', 'top', 'Interpreter', 'none', varargin{:});
    txt(0, tag, 'Color', accent);
    lines = [names, what];
    for i = 1:numel(lines)
        if i <= numel(names), fw = 'bold'; else, fw = 'normal'; end
        txt(0.26 * i, lines{i}, 'FontWeight', fw);
    end
end
hd = @(k, tag, names, what) heading(ov, W, H, cell_left(k), cell_top(k), tag, names, what, accent);
function overlay_image(ax, rgb, mask, c)
    im = rgb; for ch = 1:3, p = im(:, :, ch); p(mask) = 0.35 * p(mask) + 0.65 * c(ch); im(:, :, ch) = p; end
    image(ax, im); axis(ax, 'image', 'off');
end
function caption(ax, s)
    text(ax, 112, 232, s, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND);
end

%% 1 luminance distribution
k = 1; hd(k, 'Object: luminance', {'Y_mean, Y_med, Y_max, Y_sd,', 'Y_skew, Y_kurt, rms_contrast'}, {'linear luminance over the object''s pixels;', 'RMS contrast = SD / mean'});
[l, tp, w, h] = plot_box(k); ax = ax_cm(l - 0.45, tp, w, h);
y = Y(obj); histogram(ax, y, 40, 'FaceColor', light, 'EdgeColor', 'none', 'Normalization', 'probability');
yl = ylim(ax); ylim(ax, [0 yl(2) * 1.35]);
plot(ax, mean(y) * [1 1], [0 yl(2) * 0.72], '-', 'Color', figstyle.INK, 'LineWidth', 0.8); plot(ax, median(y) * [1 1], [0 yl(2) * 0.72], ':', 'Color', figstyle.INK, 'LineWidth', 0.8);
text(ax, 0.02, yl(2) * 1.3, {'mean: solid line', 'median: dotted line', 'max: triangle'}, small{:}, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
plot(ax, max(y), yl(2) * 0.08, 'v', 'MarkerFaceColor', accent, 'MarkerEdgeColor', 'none', 'MarkerSize', 4);
figstyle.set_label(ax, 'x', 'Luminance of the pixels'); figstyle.set_label(ax, 'y', 'Fraction'); ax.YTick = [];
figstyle.set_ticks(ax, 'x', 0:0.5:1, '%.1f');

%% 2 highlights
k = 2; hd(k, 'Object: highlights', {'hl_frac, hl_contrast'}, {'fraction of the object brighter than half', 'its maximum; RMS contrast of the top 1%'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
overlay_image(ax, rgb, obj & Y > 0.5 * max(Y(obj)), accent);
caption(ax, sprintf('Y > max / 2: %.0f%% of the object', 100 * mean(Y(obj) > 0.5 * max(Y(obj)))));

%% 3 dark pixels
k = 3; hd(k, 'Object: dark pixels', {'dark_frac'}, {'fraction of the object darker than', 'a fifth of its mean luminance'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
overlay_image(ax, rgb, obj & Y < 0.2 * mean(Y(obj)), accent);
caption(ax, sprintf('Y < mean / 5: %.0f%% of the object', 100 * mean(Y(obj) < 0.2 * mean(Y(obj)))));

%% 4 edge energy
k = 4; hd(k, 'Object: edges', {'edge_energy'}, {'mean luminance gradient over the', 'object, outline band left out'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, repmat(min(gmag / prctile(gmag(:), 99), 1), 1, 1, 3)); axis(ax, 'image', 'off');
caption(ax, '|gradient of Y|');

%% 5 chroma
k = 5; hd(k, 'Object: color', {'chroma_mean, chroma_sd'}, {'CIELAB chroma of the object''s pixels,', 'relative to the display white'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, repmat(min(chroma / prctile(chroma(obj), 99), 1) .* obj, 1, 1, 3)); axis(ax, 'image', 'off');
caption(ax, 'chroma');

%% 6 area
k = 6; hd(k, 'Object: area', {'area'}, {'fraction of the crop covered by', 'the object: its distance from the eye'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
rgb0 = im2double(imread(fullfile(ex_dir, 'example_object.png'))); Y0 = 0.2126 * rgb0(:, :, 1) + 0.7152 * rgb0(:, :, 2) + 0.0722 * rgb0(:, :, 3); obj0 = Y0 > 0.01;
overlay_image(ax, 0.75 + 0 * rgb0, obj0, accent);
caption(ax, sprintf('%.0f%% of the crop', 100 * mean(obj0(:))));

%% 7 frame difference
k = 7; hd(k, 'Object: motion', {'diff_energy'}, {'mean absolute change of luminance', 'between consecutive sampled frames'});
l = cell_left(k) + 0.1; tp = cell_top(k) + head_h; s3 = (cw - 0.55) / 3;
for i = 1:3
    ax = img_cm(l + (i - 1) * (s3 + 0.1), tp + (icon_h - s3) / 2 - 0.2, s3, s3);
    switch i
        case 1, image(ax, c0); caption(ax, 'frame t');
        case 2, image(ax, c1); caption(ax, 'frame t + 1');
        case 3, image(ax, repmat(min(abs(Yc1 - Yc0) / 0.3, 1), 1, 1, 3)); caption(ax, '|difference|');
    end
    axis(ax, 'image', 'off');
end

%% 8 the surround
k = 8; hd(k, 'Surround', {'bg_Y_mean, bg_Y_sd, bg_Y_max,', 'bg_rms_contrast, bg_chroma_mean'}, {'luminance and chroma', 'outside the object'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
overlay_image(ax, patch_rgb, ~pmask, accent);
caption(ax, sprintf('surround Y %.2f', mean(PY(~pmask))));

%% 9 summaries over the trial
k = 9; hd(k, 'Over the trial: summaries', {'*_mean, *_sd, *_max,', '*_late_minus_early'}, {'each per-frame quantity is a time series;', 'its mean, SD, max and second minus first half'});
[l, tp, w, h] = plot_box(k); ax = ax_cm(l, tp, w, h);
fill(ax, [t; flipud(t)]', [mean(v) + std(v) * ones(size(t)); flipud(mean(v) - std(v) * ones(size(t)))]', light, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
plot(ax, t, v, '-', 'Color', accent, 'LineWidth', figstyle.LW_LINE);
yline(ax, mean(v), '-', 'Color', figstyle.INK, 'LineWidth', 0.6);
hh = floor(numel(v) / 2);
plot(ax, [t(1) t(hh)], mean(v(1:hh)) * [1 1], '--', 'Color', figstyle.GRAY_TEXT, 'LineWidth', 0.8);
plot(ax, [t(hh + 1) t(end)], mean(v(hh + 1:end)) * [1 1], '--', 'Color', figstyle.GRAY_TEXT, 'LineWidth', 0.8);
[vm, im] = max(v); plot(ax, t(im), vm, 'o', 'MarkerFaceColor', accent, 'MarkerEdgeColor', figstyle.INK, 'MarkerSize', 4, 'LineWidth', 0.5);
yl = [min(v) - 0.12, max(v) + 0.08]; ylim(ax, yl);
yr = yl(2) - yl(1);
text(ax, 0.04 * t(end), yl(1) + 0.19 * yr, 'band: mean ± SD', small{:}, 'Color', figstyle.GRAY_TEXT, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
text(ax, 0.04 * t(end), yl(1) + 0.07 * yr, 'dashed: half means', small{:}, 'Color', figstyle.GRAY_TEXT, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
xlim(ax, [0 t(end)]); figstyle.set_label(ax, 'x', 'Time [s]'); figstyle.set_label(ax, 'y', 'RMS contrast');
figstyle.set_ticks(ax, 'x', 0:1:2, '%.0f');

%% 10 autocorrelation
k = 10; hd(k, 'Over the trial: autocorrelation', {'Y_mean_ac1, rms_contrast_ac1,', 'hl_frac_ac1, obj_bg_lum_ratio_ac1'}, {'lag-one autocorrelation of the series:', 'how smoothly the image changed'});
[l, tp, w, h] = plot_box(k); ax = ax_cm(l, tp, w, h);
plot(ax, v(1:end - 1), v(2:end), 'o', 'MarkerSize', 2.5, 'MarkerFaceColor', light, 'MarkerEdgeColor', 'none');
r = corr(v(1:end - 1), v(2:end));
figstyle.unity_line(ax);
figstyle.stat_text(ax, sprintf('r = %.2f', r), 'upper left');
figstyle.set_label(ax, 'x', 'Value at frame t'); figstyle.set_label(ax, 'y', 'Value at frame t + 1');
lim = [min(v) max(v)] + [-0.02 0.02]; xlim(ax, lim); ylim(ax, lim);
figstyle.set_ticks(ax, 'x', 0.2:0.2:0.8, '%.1f'); figstyle.set_ticks(ax, 'y', 0.2:0.2:0.8, '%.1f');

%% 11 time windows
k = 11; hd(k, 'Time windows', {'*_first2s, *_early_half, *_late_half'}, {'the same statistics from the first 2 s,', 'the first half or the second half only'});
[l, tp, w, h] = plot_box(k); ax = ax_cm(l, tp, w, h);
yl = [min(v) - 0.05, max(v) + 0.12];
fill(ax, [0 2 2 0], yl([1 1 2 2]), light, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
fill(ax, [t(hh + 1) t(end) t(end) t(hh + 1)], yl([1 1 2 2]), col.ramp(3, :), 'EdgeColor', 'none', 'FaceAlpha', 0.35);
plot(ax, t, v, '-', 'Color', accent, 'LineWidth', figstyle.LW_LINE);
text(ax, 0.05, yl(2), 'first 2 s', small{:}, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
text(ax, t(end), yl(2), 'second half ', small{:}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
xlim(ax, [0 t(end)]); ylim(ax, yl); figstyle.set_label(ax, 'x', 'Time [s]'); figstyle.set_label(ax, 'y', 'RMS contrast');
figstyle.set_ticks(ax, 'x', 0:1:2, '%.0f');

%% 12 object against surround
k = 12; hd(k, 'Object against surround', {'obj_bg_lum_ratio,', 'obj_bg_max_ratio,', 'obj_bg_contrast_ratio,', 'obj_bg_chroma_ratio'}, {'log ratios, object over surround'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, patch_rgb); axis(ax, 'image', 'off'); hold(ax, 'on');
B = bwboundaries(pmask); for i = 1:numel(B), plot(ax, B{i}(:, 2), B{i}(:, 1), '-', 'Color', col.ramp(5, :), 'LineWidth', 0.8); end
caption(ax, sprintf('object Y %.2f / surround Y %.2f', mean(PY(pmask)), mean(PY(~pmask))));

%% 13-16 optic-flow cues (Doerschner et al., 2011) on one frame pair
Pf = readtable(fullfile(ex_dir, 'flow_example_points.csv'));
Df = readmatrix(fullfile(ex_dir, 'flow_example_div.csv'));
g0 = im2double(imread(fullfile(ex_dir, 'flow_example_g0.png')));
gray_rgb = repmat(g0, 1, 1, 3);
pt_args = {'MarkerSize', 2.2, 'LineStyle', 'none'};

k = 13; hd(k, 'Motion: coverage', {'coverage, coverage_change'}, {'fraction of object points tracked over', '34 ms; its % decrease over 68 ms'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, gray_rgb); axis(ax, 'image', 'off'); hold(ax, 'on');
plot(ax, Pf.x(~Pf.ok1), Pf.y(~Pf.ok1), 'o', 'MarkerFaceColor', [0.55 0.55 0.55], 'MarkerEdgeColor', 'none', pt_args{:});
plot(ax, Pf.x(Pf.ok1 == 1), Pf.y(Pf.ok1 == 1), 'o', 'MarkerFaceColor', light, 'MarkerEdgeColor', 'none', pt_args{:});
caption(ax, sprintf('tracked %.0f%% (34 ms), %.0f%% (68 ms)', 100 * mean(Pf.ok1), 100 * mean(Pf.ok2)));

k = 14; hd(k, 'Motion: divergence', {'div_frac, div_mean_abs'}, {'divergence of the dense flow field over', 'the object: fraction > 0.5, mean |div|'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
A = abs(Df); A(isnan(A)) = 0; A = min(A / 0.6, 1);
im = gray_rgb * 0.8; for ch = 1:3, c = im(:, :, ch); c(~isnan(Df)) = (1 - A(~isnan(Df))) * 0.8 .* g0(~isnan(Df)) + A(~isnan(Df)) * accent(ch); im(:, :, ch) = c; end
image(ax, im); axis(ax, 'image', 'off');
caption(ax, sprintf('|div| > 0.5: %.0f%% of object pixels', 100 * mean(abs(Df(~isnan(Df))) > 0.5)));

k = 15; hd(k, 'Motion: 3-D shape reliability', {'sfm_inlier_ratio'}, {'fraction of flow vectors consistent with', 'a rigid 3-D motion (fundamental matrix)'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, gray_rgb); axis(ax, 'image', 'off'); hold(ax, 'on');
sc = 4;
kp = Pf.keep == 1;
quiver(ax, Pf.x(kp & Pf.inlier == 0), Pf.y(kp & Pf.inlier == 0), sc * Pf.u(kp & Pf.inlier == 0), sc * Pf.v(kp & Pf.inlier == 0), 0, 'Color', [0.55 0.55 0.55], 'LineWidth', 0.5, 'MaxHeadSize', 0.6);
quiver(ax, Pf.x(kp & Pf.inlier == 1), Pf.y(kp & Pf.inlier == 1), sc * Pf.u(kp & Pf.inlier == 1), sc * Pf.v(kp & Pf.inlier == 1), 0, 'Color', light, 'LineWidth', 0.5, 'MaxHeadSize', 0.6);
caption(ax, sprintf('inliers %.0f%% (blue), outliers (gray)', 100 * mean(Pf.inlier(kp))));

k = 16; hd(k, 'Motion: speed', {'flow_speed'}, {'mean magnitude of the tracked', 'vectors [px per 34 ms]'});
[l, tp, w, h] = img_box(k); ax = img_cm(l, tp, w, h);
image(ax, gray_rgb); axis(ax, 'image', 'off'); hold(ax, 'on');
ok = Pf.ok1 == 1;
quiver(ax, Pf.x(ok), Pf.y(ok), sc * Pf.u(ok), sc * Pf.v(ok), 0, 'Color', light, 'LineWidth', 0.5, 'MaxHeadSize', 0.6);
caption(ax, sprintf('mean speed %.1f px', mean(hypot(Pf.u(ok), Pf.v(ok)))));

figstyle.save(fig, fullfile(project_root, 'figs', 'figA1'));
close(fig);

function [out, scale, rr, cc] = local_zoom(rgb)
% the object crops leave the object small in a common canvas; enlarge the
% region around the object (bounding box plus a fifth of its size) for display
Y = 0.2126 * rgb(:, :, 1) + 0.7152 * rgb(:, :, 2) + 0.0722 * rgb(:, :, 3);
[r, c] = find(Y > 0.01); r0 = min(r); r1 = max(r); c0 = min(c); c1 = max(c);
side = round(max(r1 - r0, c1 - c0) * 1.4); cy = round((r0 + r1) / 2); cx = round((c0 + c1) / 2);
rr = max(1, cy - round(side / 2)):min(size(Y, 1), cy - round(side / 2) + side - 1);
cc = max(1, cx - round(side / 2)):min(size(Y, 2), cx - round(side / 2) + side - 1);
out = imresize(rgb(rr, cc, :), [224 224], 'nearest'); scale = 224 / side;
end
