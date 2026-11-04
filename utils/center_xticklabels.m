function center_xticklabels(ax, labels, y_cm)
% CENTER_XTICKLABELS  Multi-line x tick labels centered under their ticks.
%   MATLAB left-aligns the lines of a multi-line tick label; this replaces the
%   tick labels with centered text objects. labels: cell array, one per tick,
%   each a cell array of lines or a string with '\newline'. y_cm: distance
%   below the axis in cm (default 0.12).
if nargin < 3 || isempty(y_cm), y_cm = 0.12; end
xt = ax.XTick; ax.XTickLabel = repmat({''}, 1, numel(xt));
yl = ylim(ax); u = ax.Units; ax.Units = 'centimeters'; h_cm = ax.Position(4); ax.Units = u;
y = yl(1) - y_cm / h_cm * (yl(2) - yl(1));
for k = 1:numel(xt)
    l = labels{k};
    if ischar(l) || isstring(l), l = strsplit(char(l), '\newline'); end
    text(ax, xt(k), y, l, 'FontName', figstyle.FONT, 'FontSize', figstyle.FS_TICK, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Clipping', 'off');
end
end
