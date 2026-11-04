function box_plot(ax, x, v, color, ~)
% BOX_PLOT  MATLAB's compact box plot (boxplot(..., 'PlotStyle', 'compact'))
%   of one group at position x, in the given colour: a thick line between the
%   quartiles, the median as a dot, thin whiskers and outliers as small dots.
%   Draws into ax without disturbing its ticks and limits.
xl = xlim(ax); yl = ylim(ax); xt = ax.XTick; xtl = ax.XTickLabel;
h = boxplot(ax, v(:), 'Positions', x, 'PlotStyle', 'compact', 'Colors', color, 'Symbol', '.', 'Jitter', 0, 'Labels', {''});
set(findobj(h, 'Tag', 'Box'), 'LineWidth', 5);
set(findobj(h, 'Tag', 'Whisker'), 'LineWidth', figstyle.LW_THIN, 'LineStyle', '-', 'Color', color);
set(findobj(h, 'Tag', 'MedianOuter'), 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', figstyle.INK, 'LineWidth', 0.5);
set(findobj(h, 'Tag', 'MedianInner'), 'MarkerSize', 2.2, 'MarkerFaceColor', figstyle.INK, 'MarkerEdgeColor', figstyle.INK);
set(findobj(h, 'Tag', 'Outliers'), 'MarkerSize', 5, 'MarkerEdgeColor', color);
xlim(ax, xl); ylim(ax, yl); ax.XTick = xt; ax.XTickLabel = xtl; ax.Box = 'off';
end
