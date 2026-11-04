classdef figstyle
% FIGSTYLE  The figure style shared by every figure script of this repository.
%
%   Sizes, fonts, line widths, colors and export settings in one place. All
%   methods are static, so the file only needs to be on the MATLAB path. Every
%   dimension is the PRINTED size: build the figure at its final size and never
%   rescale it afterwards.
%
%   Usage:
%     fig = figstyle.figure(2, 5.2);                  % 2-column width (17.8 cm), 5.2 cm tall
%     axs = figstyle.subplots(fig, 1, 3, ...          % 1 row x 3 columns; margins in normalized units
%               struct('left',0.075,'right',0.985,'bottom',0.20,'top',0.84,'wspace',0.95));
%     h = figstyle.plot_mean(axs(1), x, y, err, color);   % mean +/- error, black-edged markers
%     figstyle.set_ticks(axs(1), 'y', 0:0.2:1, '%.1f');   % fixed ticks with consistent decimals
%     figstyle.set_label(axs(1), 'y', 'Perceived milkiness');  % wraps to 2 lines if too long
%     figstyle.legend_outside(axs(1), h, 'Lighting');
%     figstyle.panel_labels(fig, axs);                 % (a)(b)(c) at the top-left of each panel
%     figstyle.save(fig, 'fig4');                      % fig4.pdf (vector) + fig4.png (600 dpi)
%
%   Requires R2019b or later (yline/xline). exportgraphics is deliberately not used
%   because it trims the margins and changes the printed size.

properties (Constant)
    % ---- 1. Dimensions (cm) ----
    WIDTH_1COL = 8.9;      % Nature: 89 mm
    WIDTH_2COL = 17.8;     % Nature: 183 mm (17.8 cm = 7.0 in is our standard)
    MAX_HEIGHT = 24.0;     % Nature page depth 247 mm, leaving room for the caption

    % ---- 2. Fonts (pt) ----
    FONT      = 'Arial';
    FS_PANEL  = 10;        % panel labels (a) (b) ..., bold
    FS_LABEL  = 7;         % axis labels, legend titles, panel headings
    FS_TICK   = 6;         % tick labels
    FS_LEGEND = 6;         % legend entries, annotations, callout labels
    FS_SMALL  = 5;         % absolute minimum (Nature's minimum)

    % ---- 3. Lines and markers (pt) ----
    LW_AXIS   = 0.6;       % axis lines and ticks
    LW_LINE   = 1.0;       % data lines
    LW_BOLD   = 1.6;       % emphasized mean lines
    LW_THIN   = 0.5;       % individual data, callout lines, marker edges
    LW_ERR    = 0.6;       % error bars
    MS        = 5.5;       % marker diameter
    MS_BIG    = 6.5;
    TICK_PT   = 2.5;       % tick length

    % ---- 4. Panel-label offsets (cm): PAD_X to the left and PAD_Y above the panel contour ----
    PAD_X     = 0.1;
    PAD_Y     = 0.15;

    % ---- 5. Colors (RGB, 0-1) ----
    BG         = [0.97 0.97 0.97];           % plot background
    GRID_MAJOR = [1 1 1];                    % major grid: solid white
    GRID_MINOR = [0.863 0.863 0.863];        % minor grid: dotted light gray (#DCDCDC)
    INK        = [0 0 0];
    GRAY_TEXT  = [0.333 0.333 0.333];        % #555555
    GRAY_LINE  = [0.5 0.5 0.5];              % #808080

    % Ordered factors -> single-hue light-to-dark ramps (5 steps)
    SEQ_BLUE = [141 189 236; 108 148 196; 76 108 164; 44 68 124; 12 28 84] / 255;   % e.g. lighting, backlit -> frontlit
    SEQ_TEAL = [124 204 188; 92 172 156; 60 132 124; 28 92 84; 4 52 52] / 255;      % e.g. concentration, low -> high
    SEQ_RED  = [244 169 169; 229 115 115; 212 66 66; 165 34 34; 94 13 13] / 255;

    % Nominal categories -> four desaturated colors (pink, teal, olive, gray) and darker versions for text
    CAT      = [204 140 188; 92 172 156; 196 188 100; 160 160 160] / 255;
    CAT_DARK = [139 47 107; 12 84 68; 138 128 32; 85 85 85] / 255;

    % Fixed labels such as image regions (gray, blue, red)
    ROI = [128 128 128; 28 84 164; 220 68 92] / 255;
end

methods (Static)

    % ==================================================================
    % Figure and axes
    % ==================================================================
    function fig = figure(cols, height_cm, width_cm)
        % Create a figure at the printed width (1 or 2 columns).
        if nargin < 1 || isempty(cols), cols = 2; end
        if nargin < 3 || isempty(width_cm)
            if cols == 1, w = figstyle.WIDTH_1COL; else, w = figstyle.WIDTH_2COL; end
        else
            w = width_cm;
        end
        if nargin < 2 || isempty(height_cm), height_cm = w * 0.6; end
        if height_cm > figstyle.MAX_HEIGHT
            warning('figstyle:height', 'height %.1f cm exceeds MAX_HEIGHT = %.1f cm', height_cm, figstyle.MAX_HEIGHT);
        end
        fig = figure('Units', 'centimeters', 'Color', 'w', 'Renderer', 'painters', ...
                     'InvertHardcopy', 'off', ...        % keep the gray axes background when printing
                     'Visible', 'off');                  % never pop up a window; the files are the output
        pos = fig.Position; fig.Position = [pos(1) pos(2) w height_cm];
        fig.PaperUnits = 'centimeters';
        fig.PaperSize = [w height_cm];
        fig.PaperPositionMode = 'manual';
        fig.PaperPosition = [0 0 w height_cm];
        set(fig, 'DefaultAxesFontName', figstyle.FONT, 'DefaultTextFontName', figstyle.FONT, ...
                 'DefaultAxesFontSize', figstyle.FS_TICK, 'DefaultTextFontSize', figstyle.FS_LEGEND, ...
                 'DefaultLegendFontName', figstyle.FONT, 'DefaultLegendFontSize', figstyle.FS_LEGEND, ...
                 'DefaultLegendBox', 'off', 'DefaultLineLineWidth', figstyle.LW_LINE);
    end

    function ax = axes(fig, pos)
        % Create styled axes at normalized position pos = [left bottom width height].
        ax = axes('Parent', fig, 'Units', 'normalized', 'Position', pos);
        figstyle.style_axis(ax);
    end

    function axs = subplots(fig, nrows, ncols, m)
        % Lay out nrows x ncols axes, following matplotlib's subplots_adjust convention.
        % m: struct with fields left, right, bottom, top (normalized figure coordinates)
        %    and wspace, hspace (gaps as a fraction of the axes width / height).
        d = struct('left', 0.08, 'right', 0.98, 'bottom', 0.18, 'top', 0.86, 'wspace', 0.5, 'hspace', 0.5);
        if nargin >= 4
            f = fieldnames(m);
            for k = 1:numel(f), d.(f{k}) = m.(f{k}); end
        end
        w = (d.right - d.left) / (ncols + (ncols - 1) * d.wspace);
        h = (d.top - d.bottom) / (nrows + (nrows - 1) * d.hspace);
        axs = gobjects(nrows, ncols);
        for r = 1:nrows
            for c = 1:ncols
                l = d.left + (c - 1) * w * (1 + d.wspace);
                b = d.top - r * h - (r - 1) * h * d.hspace;
                axs(r, c) = figstyle.axes(fig, [l b w h]);
            end
        end
    end

    function style_axis(ax)
        % Axis style: left/bottom spines only, short outward ticks, gray background,
        % white major grid, dotted minor grid.
        set(ax, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_TICK, ...
                'Color', figstyle.BG, 'Box', 'off', ...
                'XColor', figstyle.INK, 'YColor', figstyle.INK, 'LineWidth', figstyle.LW_AXIS, ...
                'TickDir', 'out', 'Layer', 'bottom', ...
                'XGrid', 'on', 'YGrid', 'on', 'GridColor', figstyle.GRID_MAJOR, 'GridAlpha', 1, ...
                'GridLineStyle', '-', ...
                'XMinorGrid', 'on', 'YMinorGrid', 'on', 'MinorGridColor', figstyle.GRID_MINOR, ...
                'MinorGridAlpha', 1, 'MinorGridLineStyle', ':', ...
                'XMinorTick', 'off', 'YMinorTick', 'off', ...
                'TitleFontSizeMultiplier', 1, 'TitleFontWeight', 'normal', ...
                'LabelFontSizeMultiplier', figstyle.FS_LABEL / figstyle.FS_TICK, ...
                'NextPlot', 'add');
        try   % R2023a and later: separate grid line widths
            ax.GridLineWidth = 0.8;
            ax.MinorGridLineWidth = 0.5;
        catch
        end
        ax.XLabel.FontSize = figstyle.FS_LABEL;
        ax.YLabel.FontSize = figstyle.FS_LABEL;
        figstyle.tick_length(ax, figstyle.TICK_PT);
    end

    function tick_length(ax, pt)
        % Set the tick length in points (MATLAB specifies it as a fraction of the longer axis).
        u = ax.Units; ax.Units = 'centimeters';
        L = max(ax.Position(3:4)) * 72 / 2.54;     % longer axis length in pt
        ax.Units = u;
        ax.TickLength = [pt / L, pt / L];
    end

    function set_ticks(ax, axis, ticks, fmt)
        % Fix the tick positions and make the number of decimals consistent.
        % fmt: a format such as '%.2f', or a cell/string array of text labels.
        % With text labels (a categorical axis) the minor grid in that direction is removed.
        ticks = ticks(:)';
        if nargin < 4, fmt = []; end
        if strcmpi(axis, 'x'), ruler = ax.XAxis; else, ruler = ax.YAxis; end
        ruler.TickValues = ticks;
        if iscell(fmt) || isstring(fmt)
            ruler.TickLabels = cellstr(fmt);
            if strcmpi(axis, 'x'), ax.XMinorGrid = 'off'; else, ax.YMinorGrid = 'off'; end
            return
        end
        if isempty(fmt)
            dec = max(arrayfun(@figstyle.decimals, ticks));
            fmt = sprintf('%%.%df', dec);
        end
        ruler.TickLabels = arrayfun(@(v) sprintf(fmt, v), ticks, 'UniformOutput', false);
        % One minor grid line at the midpoint between major ticks
        if numel(ticks) > 1
            ruler.MinorTickValues = ticks(1:end-1) + diff(ticks) / 2;
        end
    end

    function n = decimals(x)
        % Number of decimals needed to print x exactly (0.25 -> 2, 0.5 -> 1, 1 -> 0).
        s = sprintf('%.10f', x); s = regexprep(s, '0+$', ''); s = regexprep(s, '\.$', '');
        k = strfind(s, '.');
        if isempty(k), n = 0; else, n = numel(s) - k; end
    end

    function set_label(ax, axis, txt, max_lines)
        % Set an axis label; wrap it to two lines at a natural break if it is longer than the axis.
        if nargin < 4, max_lines = 2; end
        txt = char(txt);
        if strcmpi(axis, 'x'), lab = ax.XLabel; else, lab = ax.YLabel; end
        lab.String = txt; lab.FontSize = figstyle.FS_LABEL; lab.FontName = figstyle.FONT;
        if max_lines < 2 || iscell(txt) || contains(txt, newline), return; end
        drawnow;
        u = ax.Units; ax.Units = 'centimeters';
        if strcmpi(axis, 'x'), axis_len = ax.Position(3); else, axis_len = ax.Position(4); end
        ax.Units = u;
        lu = lab.Units; lab.Units = 'centimeters';
        ext = lab.Extent;                      % [x y w h] in cm (bounding box after rotation)
        lab.Units = lu;
        if strcmpi(axis, 'x'), label_len = ext(3); else, label_len = ext(4); end
        if label_len > axis_len
            lab.String = figstyle.wrap_two(txt);
        end
    end

    function out = wrap_two(txt)
        % Split text into two lines of similar length, preferring natural breaks.
        % Scoring (lower is better): max line length; -3 for breaking before a bracket
        % (units, e.g. '[ml]') or a parenthesis or a dash; -1 for breaking before a
        % preposition; +1 for ending a line with a preposition or article.
        preps = {'of', 'in', 'to', 'per', 'with', 'across', 'between', 'against', 'vs', 'for'};
        arts  = {'the', 'a', 'an', 'and', 'or'};
        words = strsplit(char(txt), ' ');
        if numel(words) < 2, out = txt; return; end
        best = 1; best_score = inf;
        for k = 2:numel(words)
            left = strjoin(words(1:k-1), ' '); right = strjoin(words(k:end), ' ');
            score = max(numel(left), numel(right));
            if any(right(1) == ['[' '(' char(8722) char(8211) '-'])
                score = score - 3;
            elseif any(strcmpi(words{k}, preps))
                score = score - 1;
            end
            if any(strcmpi(words{k-1}, [preps arts])), score = score + 1; end
            if score < best_score, best = k; best_score = score; end
        end
        out = {strjoin(words(1:best-1), ' '); strjoin(words(best:end), ' ')};
    end

    % ==================================================================
    % Reference lines
    % ==================================================================
    function h = zero_line(ax, y)
        % Baseline such as y = 0: thin black line.
        if nargin < 2, y = 0; end
        h = yline(ax, y);
        set(h, 'LineStyle', '-', 'Color', figstyle.INK, 'LineWidth', figstyle.LW_AXIS);
        try, uistack(h, 'bottom'); catch, end
    end

    function h = ref_line(ax, orientation, value, label)
        % Reference line (gray dashed) with a 6-pt label at the right end. orientation: 'h' or 'v'.
        if nargin < 4, label = ''; end
        if strcmpi(orientation, 'h')
            h = yline(ax, value, '--', label, 'Color', figstyle.GRAY_TEXT, 'LineWidth', figstyle.LW_AXIS, ...
                      'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
        else
            h = xline(ax, value, '--', label, 'Color', figstyle.GRAY_TEXT, 'LineWidth', figstyle.LW_AXIS, ...
                      'LabelOrientation', 'aligned', 'LabelVerticalAlignment', 'top');
        end
        h.FontSize = figstyle.FS_LEGEND; h.FontName = figstyle.FONT;
        try, uistack(h, 'bottom'); catch, end
    end

    function h = unity_line(ax)
        % y = x line for scatter plots (thin gray).
        lo = min(ax.XLim(1), ax.YLim(1)); hi = max(ax.XLim(2), ax.YLim(2));
        h = plot(ax, [lo hi], [lo hi], '-', 'Color', figstyle.GRAY_LINE, 'LineWidth', figstyle.LW_AXIS);
        uistack(h, 'bottom');
    end

    % ==================================================================
    % Data
    % ==================================================================
    function hm = plot_mean(ax, x, y, err, color, lw, ms, marker)
        % Mean +/- error: thin line, large black-edged markers, thin capless error bars in the
        % same color. Returns the marker handle (for the legend; set its DisplayName afterwards).
        if nargin < 4, err = []; end
        if nargin < 5 || isempty(color), color = figstyle.SEQ_BLUE(3, :); end
        if nargin < 6 || isempty(lw), lw = figstyle.LW_LINE; end
        if nargin < 7 || isempty(ms), ms = figstyle.MS; end
        if nargin < 8 || isempty(marker), marker = 'o'; end
        if ~isempty(err)
            he = errorbar(ax, x, y, err, 'LineStyle', 'none', 'Color', color, ...
                          'LineWidth', figstyle.LW_ERR, 'CapSize', 0);
            he.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        hl = plot(ax, x, y, '-', 'Color', color, 'LineWidth', lw);
        hl.Annotation.LegendInformation.IconDisplayStyle = 'off';
        hm = plot(ax, x, y, 'LineStyle', 'none', 'Marker', marker, 'MarkerSize', ms, ...
                  'MarkerFaceColor', color, 'MarkerEdgeColor', figstyle.INK, 'LineWidth', figstyle.LW_THIN);
    end

    function hs = plot_individuals(ax, x, Y, color, lw, alpha)
        % Individual data: thin semi-transparent lines connecting conditions. Y is (n_subjects x n_x).
        % Transparency is emulated by blending with the background color, which is robust in PDF output.
        if nargin < 5 || isempty(lw), lw = figstyle.LW_THIN; end
        if nargin < 6 || isempty(alpha), alpha = 0.35; end
        c = alpha * color + (1 - alpha) * figstyle.BG;
        hs = gobjects(size(Y, 1), 1);
        for i = 1:size(Y, 1)
            hs(i) = plot(ax, x, Y(i, :), '-', 'Color', c, 'LineWidth', lw);
            hs(i).Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
    end

    function hs = strip(ax, x_center, values, color, width, size_pt, alpha, seed)
        % Strip plot: small edgeless dots jittered horizontally around x_center.
        if nargin < 5 || isempty(width), width = 0.35; end
        if nargin < 6 || isempty(size_pt), size_pt = 2.0; end
        if nargin < 7 || isempty(alpha), alpha = 1.0; end
        if nargin < 8 || isempty(seed), seed = 0; end
        rs = RandStream('mt19937ar', 'Seed', seed);
        values = values(:);
        xs = x_center + (rand(rs, numel(values), 1) * 2 - 1) * width;
        hs = scatter(ax, xs, values, size_pt^2, color, 'filled', 'MarkerEdgeColor', 'none', ...
                     'MarkerFaceAlpha', alpha);
        hs.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end

    function h = mean_bar(ax, x_center, value, color, width, lw)
        % Short horizontal line marking a group mean on a strip plot.
        if nargin < 4 || isempty(color), color = figstyle.INK; end
        if nargin < 5 || isempty(width), width = 0.35; end
        if nargin < 6 || isempty(lw), lw = 1.2; end
        h = plot(ax, [x_center - width, x_center + width], [value value], '-', 'Color', color, 'LineWidth', lw);
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end

    function h = bars(ax, x, y, color, width)
        % Bar chart without edges.
        if nargin < 5 || isempty(width), width = 0.7; end
        h = bar(ax, x, y, width, 'FaceColor', color, 'EdgeColor', 'none');
    end

    % ==================================================================
    % Annotations and legends
    % ==================================================================
    function [ht, hl] = callout(ax, xy, txt, xytext, color, ha, va)
        % Callout: thin gray straight line (no arrowhead) from xy to xytext plus a 6-pt label.
        % Coordinates are in data units.
        if nargin < 5 || isempty(color), color = figstyle.INK; end
        if nargin < 6 || isempty(ha), ha = 'center'; end
        if nargin < 7 || isempty(va), va = 'bottom'; end
        hl = plot(ax, [xy(1) xytext(1)], [xy(2) xytext(2)], '-', 'Color', figstyle.GRAY_TEXT, ...
                  'LineWidth', figstyle.LW_THIN);
        hl.Annotation.LegendInformation.IconDisplayStyle = 'off';
        ht = text(ax, xytext(1), xytext(2), txt, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, ...
                  'Color', color, 'HorizontalAlignment', ha, 'VerticalAlignment', va);
    end

    function lgd = legend_outside(ax, handles, ttl, gap_cm)
        % Frameless legend to the right of the axes without shrinking the axes.
        % ttl is the legend title (7 pt).
        if nargin < 3, ttl = ''; end
        if nargin < 4 || isempty(gap_cm), gap_cm = 0.15; end
        lgd = legend(ax, handles, 'Box', 'off', 'FontName', figstyle.FONT, ...
                     'FontSize', figstyle.FS_LEGEND, 'AutoUpdate', 'off', 'Location', 'none');
        if ~isempty(ttl)
            title(lgd, ttl); lgd.Title.FontSize = figstyle.FS_LABEL; lgd.Title.FontWeight = 'normal';
        end
        drawnow;
        au = ax.Units; ax.Units = 'centimeters'; ap = ax.Position; ax.Units = au;
        lu = lgd.Units; lgd.Units = 'centimeters'; lp = lgd.Position;
        lgd.Position = [ap(1) + ap(3) + gap_cm, ap(2) + ap(4) - lp(4), lp(3), lp(4)];
        lgd.Units = lu;
    end

    function lgd = legend_top(fig, handles, labels, y_norm)
        % One-row frameless legend at the top of the figure (one legend for a grid of panels).
        if nargin < 4 || isempty(y_norm), y_norm = 0.96; end
        lgd = legend(handles, labels, 'Box', 'off', 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LEGEND, ...
                     'Orientation', 'horizontal', 'AutoUpdate', 'off', 'Location', 'none');
        drawnow;
        lgd.Units = 'normalized';
        lgd.Position(1:2) = [1 - lgd.Position(3) - 0.01, y_norm - lgd.Position(4)];
    end

    function ht = stat_text(ax, txt, loc, color)
        % Statistic such as 'R^2 = 0.85' in a corner of the panel. loc: 'upper left', 'lower right', ...
        if nargin < 3 || isempty(loc), loc = 'upper left'; end
        if nargin < 4 || isempty(color), color = figstyle.INK; end
        if contains(loc, 'left'), x = 0.04; ha = 'left'; else, x = 0.96; ha = 'right'; end
        if contains(loc, 'upper'), y = 0.96; va = 'top'; else, y = 0.04; va = 'bottom'; end
        ht = text(ax, x, y, txt, 'Units', 'normalized', 'FontName', figstyle.FONT, ...
                  'FontSize', figstyle.FS_LEGEND, 'Color', color, 'HorizontalAlignment', ha, ...
                  'VerticalAlignment', va);
    end

    function panel_title(ax, txt, color, weight)
        % Heading above a panel, e.g. 'Experiment 1 (n = 12)'. Left-aligned, 7 pt.
        if nargin < 3 || isempty(color), color = figstyle.INK; end
        if nargin < 4 || isempty(weight), weight = 'normal'; end
        t = title(ax, txt, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_LABEL, ...
                  'FontWeight', weight, 'Color', color, 'HorizontalAlignment', 'left');
        t.Units = 'normalized'; t.Position(1) = 0;
    end

    % ==================================================================
    % Panel labels
    % ==================================================================
    function hs = panel_labels(fig, axs, labels, style)
        % Place (a) (b) ... at the top-left of each panel's contour (axes + TightInset, i.e.
        % including tick labels and axis labels), offset by PAD_X to the left and PAD_Y above.
        % style: 'paren' -> (a)   'nature' -> a   'upper' -> A
        % Labels within 0.5 cm of each other are aligned to the same row / column.
        axs = axs(:)';
        if nargin < 4 || isempty(style), style = 'paren'; end
        if nargin < 3 || isempty(labels)
            labels = cell(1, numel(axs));
            for k = 1:numel(axs)
                c = char('a' + k - 1);
                switch style
                    case 'nature', labels{k} = c;
                    case 'upper',  labels{k} = upper(c);
                    otherwise,     labels{k} = ['(' c ')'];
                end
            end
        end
        drawnow;
        fu = fig.Units; fig.Units = 'centimeters'; W = fig.Position(3); H = fig.Position(4); fig.Units = fu;
        pos = zeros(numel(axs), 2);
        for k = 1:numel(axs)
            ax = axs(k);
            u = ax.Units; ax.Units = 'centimeters';
            p = ax.Position; ti = ax.TightInset;
            ax.Units = u;
            pos(k, :) = [p(1) - ti(1) - figstyle.PAD_X, p(2) + p(4) + ti(4) + figstyle.PAD_Y];
        end
        for i = 1:size(pos, 1)          % align rows and columns
            for j = 1:size(pos, 1)
                if abs(pos(i, 2) - pos(j, 2)) < 0.5, m = max(pos(i, 2), pos(j, 2)); pos(i, 2) = m; pos(j, 2) = m; end
                if abs(pos(i, 1) - pos(j, 1)) < 0.5, m = min(pos(i, 1), pos(j, 1)); pos(i, 1) = m; pos(j, 1) = m; end
            end
        end
        ov = figstyle.overlay(fig);
        hs = gobjects(1, numel(axs));
        for k = 1:numel(axs)
            hs(k) = text(ov, pos(k, 1) / W, pos(k, 2) / H, labels{k}, 'Units', 'normalized', ...
                         'FontName', figstyle.FONT, 'FontSize', figstyle.FS_PANEL, 'FontWeight', 'bold', ...
                         'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
        end
    end

    function ov = overlay(fig)
        % Invisible axes covering the whole figure, used for text in figure coordinates
        % (panel labels). Created once and reused.
        ov = findall(fig, 'Type', 'axes', 'Tag', 'figstyle-overlay');
        if isempty(ov)
            ov = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1 1], 'Visible', 'off', ...
                      'HitTest', 'off', 'XLim', [0 1], 'YLim', [0 1], 'Tag', 'figstyle-overlay', ...
                      'HandleVisibility', 'off', 'NextPlot', 'add');
        end
        uistack(ov, 'top');
    end

    % ==================================================================
    % Saving and checking
    % ==================================================================
    function ok = check_fit(fig)
        % Check that every axes (with its labels), legend and panel label lies inside the
        % canvas, and that no text is smaller than FS_SMALL. Warns on each violation.
        drawnow;
        fu = fig.Units; fig.Units = 'centimeters'; W = fig.Position(3); H = fig.Position(4); fig.Units = fu;
        ok = true;
        axs = findobj(fig, 'Type', 'axes', '-not', 'Tag', 'figstyle-overlay');
        for ax = axs'
            u = ax.Units; ax.Units = 'centimeters'; p = ax.Position; ti = ax.TightInset; ax.Units = u;
            x0 = p(1) - ti(1); y0 = p(2) - ti(2); x1 = p(1) + p(3) + ti(3); y1 = p(2) + p(4) + ti(4);
            if x0 < -0.02 || y0 < -0.02 || x1 > W + 0.02 || y1 > H + 0.02
                ok = false;
                warning('figstyle:fit', 'Axes content exceeds the canvas: [%.2f %.2f %.2f %.2f] cm vs [0 0 %.2f %.2f]. Adjust the margins; do not rescale.', x0, y0, x1, y1, W, H);
            end
        end
        lg = findobj(fig, 'Type', 'legend');
        for l = lg'
            u = l.Units; l.Units = 'centimeters'; p = l.Position; l.Units = u;
            if p(1) < 0 || p(2) < 0 || p(1) + p(3) > W || p(2) + p(4) > H
                ok = false;
                warning('figstyle:fit', 'Legend exceeds the canvas: [%.2f %.2f %.2f %.2f] cm.', p(1), p(2), p(3), p(4));
            end
        end
        ov = findall(fig, 'Type', 'axes', 'Tag', 'figstyle-overlay');
        for t = findall(ov, 'Type', 'text')'
            u = t.Units; t.Units = 'normalized'; e = t.Extent; t.Units = u;
            if e(1) < 0 || e(2) < 0 || e(1) + e(3) > 1 || e(2) + e(4) > 1
                ok = false;
                warning('figstyle:fit', 'Panel label "%s" exceeds the canvas.', strjoin(cellstr(t.String), ' | '));
            end
        end
        tx = findall(fig, 'Type', 'text', '-property', 'FontSize');
        for t = tx'
            if ~isempty(t.String) && t.FontSize < figstyle.FS_SMALL
                warning('figstyle:font', 'Text "%s" is %.1f pt (< %d pt)', strjoin(cellstr(t.String), ' '), t.FontSize, figstyle.FS_SMALL);
            end
        end
    end

    function files = save(fig, stem, do_pdf, do_png, dpi)
        % Save stem.pdf (vector) and stem.png (600 dpi) at the printed size.
        % Uses print with PaperSize = figure size; exportgraphics is avoided because it trims
        % the margins and changes the size.
        if nargin < 3 || isempty(do_pdf), do_pdf = true; end
        if nargin < 4 || isempty(do_png), do_png = true; end
        if nargin < 5 || isempty(dpi), dpi = 600; end
        figstyle.check_fit(fig);
        fu = fig.Units; fig.Units = 'centimeters'; w = fig.Position(3); h = fig.Position(4); fig.Units = fu;
        fig.PaperUnits = 'centimeters'; fig.PaperSize = [w h];
        fig.PaperPositionMode = 'manual'; fig.PaperPosition = [0 0 w h];
        files = {};
        if do_pdf
            print(fig, [stem '.pdf'], '-dpdf', '-painters');
            files{end+1} = [stem '.pdf'];
        end
        if do_png
            print(fig, [stem '.png'], '-dpng', sprintf('-r%d', dpi));
            files{end+1} = [stem '.png'];
        end
    end

    function s = describe(fig)
        % Figure size and the set of font sizes in use (for checking).
        fu = fig.Units; fig.Units = 'centimeters'; w = fig.Position(3); h = fig.Position(4); fig.Units = fu;
        tx = findall(fig, '-property', 'FontSize');
        fs = get(tx, 'FontSize'); if iscell(fs), fs = cell2mat(fs); end
        sz = unique(round(fs, 1));
        s = sprintf('%.2f x %.2f cm; font sizes used: %s', w, h, mat2str(sz'));
    end

    % ==================================================================
    % Colors
    % ==================================================================
    function c = seq(n, base)
        % n-step ramp for an ordered factor (taken from the table for n <= 5, interpolated above).
        if nargin < 2 || isempty(base), base = 'blue'; end
        switch lower(base)
            case 'teal', tbl = figstyle.SEQ_TEAL;
            case 'red',  tbl = figstyle.SEQ_RED;
            otherwise,   tbl = figstyle.SEQ_BLUE;
        end
        m = size(tbl, 1);
        if n <= m
            idx = round(linspace(1, m, n));
            c = tbl(idx, :);
        else
            c = interp1(linspace(0, 1, m), tbl, linspace(0, 1, n));
        end
    end

    function rgb = hex(h)
        % '#8DBDEC' -> [r g b] in 0-1
        h = char(h); if h(1) == '#', h = h(2:end); end
        rgb = double(sscanf(h, '%2x%2x%2x')') / 255;
    end

end
end
