function run_all()
%RUN_ALL Regenerate every figure of the paper.
%   Make this repository the current folder in MATLAB and call RUN_ALL.
%   Each fig*.m script reads the tables in data/ and results/ and writes its
%   figure into figs/. The statistics quoted in the paper are printed to the
%   Command Window as each figure is drawn.
%
%   The tables in results/ are produced by the Python pipeline in python/
%   (see README.md); they are shipped so that the figures can be rebuilt
%   without the 200 GB of frames.

figure_scripts = {'fig1_stimuli', 'fig2_model', 'fig3_behaviour', 'fig4_predictors', ...
    'fig5_statistics', 'fig6_statistics_observers', 'fig7_disagreement', 'figA1_statistics'};
for k = 1:numel(figure_scripts)
    name = figure_scripts{k};
    fprintf('\n===== %s (%d/%d) =====\n', name, k, numel(figure_scripts));
    try
        evalin('base', name);
    catch err
        fprintf(2, 'Skipped %s: %s\n', name, err.message);
    end
end
end
