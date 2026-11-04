function crop_transparent_objects(projectRoot)
% CROP_TRANSPARENT_OBJECTS
% Batch-crops the object region from frames and saves them to a parallel
% "Cropped" directory while keeping the same participant/condition/filename.
%
% Assumed input structure:
%   data/imgs/participant_005/Transparent/<condition>/<time>.png
%
% Output structure (created automatically):
%   data/imgs/participant_005/Cropped/<condition>/<time>.png
%
% Notes:
% - Condition folder names are auto-detected.
% - The timestamp part (e.g., "16.331" from "16.331.png") is auto-parsed.
% - Crops are standardized to a common canvas size (computed from the dataset)
%   and then optionally resized to a fixed final size for ML.
%
% Requirements:
% - Image Processing Toolbox is recommended (imbinarize/regionprops/etc.)
%
% Usage:
%   crop_transparent_objects;                 % run from project root
%   crop_transparent_objects("path/to/root"); % explicit root

% -----------------------------
% Resolve project root
% -----------------------------
if nargin < 1 || isempty(projectRoot)
    projectRoot = pwd;
end

CFG.rootImgsDir       = fullfile(projectRoot, "data", "imgs");
CFG.transparentFolder = "Transparent";
CFG.outputFolder      = "Cropped";
CFG.filePattern       = "*.png";

% -----------------------------
% Object detection / crop config
% -----------------------------
CFG.marginFrac     = 0.03;  % extra margin around detected object bbox
CFG.minObjectArea  = 80;    % remove tiny blobs (pixels)
CFG.closeRadius    = 1;     % morphology closing radius (pixels)
CFG.threshFloor    = 0.02;  % minimum threshold after background subtraction

% Common canvas size selection across ALL images:
CFG.canvasMode        = "percentile"; % "percentile" or "max"
CFG.canvasPercentile  = 99;           % used if canvasMode == "percentile"
CFG.forceSquareCanvas = true;         % force square output canvas

% Final size for ML (set [] to keep canvas size as-is):
CFG.finalSize = [224 224];  % e.g., [224 224], or [] for no resizing

CFG.verbose = true;

% -----------------------------
% Pass 1: scan dataset and detect bbox for each image
% -----------------------------
if CFG.verbose
    fprintf("Scanning dataset under: %s\n", CFG.rootImgsDir);
end

[imgList, bboxList] = scanDatasetForBBoxes(CFG);

if isempty(imgList)
    warning("No images found. Check your folder structure and CFG settings.");
    return;
end

% -----------------------------
% Decide a common canvas size
% -----------------------------
[canvasH, canvasW] = chooseCanvasSize(bboxList, CFG);

if CFG.verbose
    fprintf("Common canvas size (HxW): %d x %d\n", canvasH, canvasW);
    if ~isempty(CFG.finalSize)
        fprintf("Final resize size (HxW): %d x %d\n", CFG.finalSize(1), CFG.finalSize(2));
    else
        fprintf("Final resize: disabled\n");
    end
end

% -----------------------------
% Pass 2: crop/pad/resize and write
% -----------------------------
processAndWriteCrops(imgList, bboxList, [canvasH canvasW], CFG);

if CFG.verbose
    fprintf("Done.\n");
end

end % main function

% =======================================================================
% Helper: scan dataset and compute object bounding boxes
% =======================================================================
function [imgList, bboxList] = scanDatasetForBBoxes(CFG)

imgList  = struct('inPath', {}, 'outPath', {}, 'participant', {}, 'condition', {}, ...
                  'timeValue', {}, 'imgSize', {});
bboxList = zeros(0,4);

pDirs = dir(fullfile(CFG.rootImgsDir, "participant_*"));
pDirs = pDirs([pDirs.isdir]);

for ip = 1:numel(pDirs)
    participantName = string(pDirs(ip).name);
    transparentDir  = fullfile(CFG.rootImgsDir, participantName, CFG.transparentFolder);

    if ~isfolder(transparentDir)
        continue;
    end

    condDirs = dir(transparentDir);
    condDirs = condDirs([condDirs.isdir]);
    condDirs = condDirs(~ismember({condDirs.name}, {'.','..'}));

    for ic = 1:numel(condDirs)
        conditionName = string(condDirs(ic).name);
        condPath      = fullfile(transparentDir, conditionName);

        imgFiles = dir(fullfile(condPath, CFG.filePattern));
        for ii = 1:numel(imgFiles)
            inPath = fullfile(condPath, imgFiles(ii).name);

            % Parse timestamp (e.g., "16.331" from "16.331.png")
            [~, stem, ~] = fileparts(imgFiles(ii).name);
            timeValue = parseTimeFromName(stem);

            % Read image and detect object bbox
            I = imread(inPath);
            bbox = detectObjectBBox(I, CFG);  % [x y w h], double

            % Build output path (keep condition and filename)
            outDir  = fullfile(CFG.rootImgsDir, participantName, CFG.outputFolder, conditionName);
            outPath = fullfile(outDir, imgFiles(ii).name);

            % Append
            s.inPath      = inPath;
            s.outPath     = outPath;
            s.participant = participantName;
            s.condition   = conditionName;
            s.timeValue   = timeValue;
            s.imgSize     = [size(I,1), size(I,2)];

            imgList(end+1,1) = s; %#ok<AGROW>
            bboxList(end+1,:) = bbox; %#ok<AGROW>
        end
    end
end

if CFG.verbose
    fprintf("Found %d images.\n", numel(imgList));
end

end

% =======================================================================
% Helper: choose a common canvas size from bbox list
% =======================================================================
function [canvasH, canvasW] = chooseCanvasSize(bboxList, CFG)

w = bboxList(:,3);
h = bboxList(:,4);

switch lower(string(CFG.canvasMode))
    case "max"
        canvasW = ceil(max(w));
        canvasH = ceil(max(h));

    case "percentile"
        canvasW = ceil(prctile(w, CFG.canvasPercentile));
        canvasH = ceil(prctile(h, CFG.canvasPercentile));

    otherwise
        error("Unknown CFG.canvasMode: %s", CFG.canvasMode);
end

% Avoid degenerate sizes
canvasW = max(canvasW, 2);
canvasH = max(canvasH, 2);

% Optionally force square canvas
if CFG.forceSquareCanvas
    side = max(canvasW, canvasH);
    canvasW = side;
    canvasH = side;
end

end

% =======================================================================
% Helper: crop/pad/resize and write images
% =======================================================================
function processAndWriteCrops(imgList, bboxList, canvasHW, CFG)

canvasH = canvasHW(1);
canvasW = canvasHW(2);

for i = 1:numel(imgList)
    inPath  = imgList(i).inPath;
    outPath = imgList(i).outPath;

    % Ensure output directory exists
    outDir = fileparts(outPath);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    I = imread(inPath);

    % Ensure we preserve channels (RGB vs grayscale) consistently per image
    % If RGBA, drop alpha channel
    if ndims(I) == 3 && size(I,3) == 4
        I = I(:,:,1:3);
    end

    bbox = bboxList(i,:);

    % Crop window centered on detected bbox, with padding if needed
    cx = bbox(1) + bbox(3)/2;
    cy = bbox(2) + bbox(4)/2;

    Ic = cropCenteredWithPadding(I, cx, cy, canvasH, canvasW);

    % Optional final resize to fixed ML input size
    if ~isempty(CFG.finalSize)
        Ic = imresize(Ic, CFG.finalSize);
    end

    imwrite(Ic, outPath);

    if CFG.verbose && mod(i, 500) == 0
        fprintf("Processed %d / %d\n", i, numel(imgList));
    end
end

end

% =======================================================================
% Helper: detect object bounding box from a single image
% =======================================================================
function bbox = detectObjectBBox(I, CFG)
% Robust simple detection for empty/black background:
% "anything not near-black" is considered object.

% Drop alpha if present
if ndims(I) == 3 && size(I,3) == 4
    I = I(:,:,1:3);
end

% Convert to grayscale double [0,1]
if ndims(I) == 3
    Ig = rgb2gray(I);
else
    Ig = I;
end
Ig = im2double(Ig);

H = size(Ig,1);
W = size(Ig,2);

% Simple threshold: non-black pixels
BW = Ig > CFG.threshFloor;

% Remove tiny noise
BW = bwareaopen(BW, CFG.minObjectArea);

% Light morphology to connect small gaps
if CFG.closeRadius > 0
    se = strel('disk', CFG.closeRadius, 0);
    BW = imclose(BW, se);
end

% Keep largest component (the object)
if any(BW(:))
    BW = keepLargestComponent(BW);
    stats = regionprops(BW, 'BoundingBox');
    bbox = stats(1).BoundingBox;
else
    % Fallback: nothing detected
    bbox = [1 1 W H];
end

% Add margin and clip
bbox = addMarginToBBox(bbox, [H W], CFG.marginFrac);
end

% =======================================================================
% Helper: crop a fixed-size window centered at (cx, cy) with padding
% =======================================================================
function out = cropCenteredWithPadding(I, cx, cy, outH, outW)

H = size(I,1);
W = size(I,2);
if ndims(I) == 2
    C = 1;
else
    C = size(I,3);
end

% Compute desired crop box in source coordinates
x1 = round(cx - outW/2);
y1 = round(cy - outH/2);
x2 = x1 + outW - 1;
y2 = y1 + outH - 1;

% Clamp to source image
sx1 = max(1, x1);
sy1 = max(1, y1);
sx2 = min(W, x2);
sy2 = min(H, y2);

% Destination placement indices
dx1 = sx1 - x1 + 1;
dy1 = sy1 - y1 + 1;
dx2 = dx1 + (sx2 - sx1);
dy2 = dy1 + (sy2 - sy1);

% Create output canvas (zero = black background)
if C == 1
    out = zeros(outH, outW, 'like', I);
    out(dy1:dy2, dx1:dx2) = I(sy1:sy2, sx1:sx2);
else
    out = zeros(outH, outW, C, 'like', I);
    out(dy1:dy2, dx1:dx2, :) = I(sy1:sy2, sx1:sx2, :);
end

end

% =======================================================================
% Helper: parse numeric timestamp from filename stem
% =======================================================================
function t = parseTimeFromName(stem)
% stem is filename without extension, e.g. "16.331"
% This returns NaN if parsing fails.

t = str2double(stem);
if ~isfinite(t)
    % Fallback: extract first number pattern
    tok = regexp(stem, '(\d+(\.\d+)?)', 'match', 'once');
    if isempty(tok)
        t = NaN;
    else
        t = str2double(tok);
    end
end

end

% =======================================================================
% Helper: add margin to bbox and clip to image bounds
% =======================================================================
function bbox2 = addMarginToBBox(bbox, imgHW, marginFrac)
% bbox: [x y w h], imgHW: [H W]

H = imgHW(1);
W = imgHW(2);

x = bbox(1); y = bbox(2); bw = bbox(3); bh = bbox(4);

mx = bw * marginFrac;
my = bh * marginFrac;

x1 = x - mx;
y1 = y - my;
x2 = x + bw + mx;
y2 = y + bh + my;

% Clip to image bounds (1-based)
x1 = max(1, x1);
y1 = max(1, y1);
x2 = min(W, x2);
y2 = min(H, y2);

bbox2 = [x1, y1, (x2 - x1), (y2 - y1)];

% Avoid zeros
bbox2(3) = max(bbox2(3), 1);
bbox2(4) = max(bbox2(4), 1);

end

% =======================================================================
% Helper: keep largest connected component (compat-friendly)
% =======================================================================
function BW2 = keepLargestComponent(BW)

if exist('bwareafilt', 'file') == 2
    BW2 = bwareafilt(BW, 1);
    return;
end

% Fallback if bwareafilt is not available
CC = bwconncomp(BW);
if CC.NumObjects == 0
    BW2 = BW;
    return;
end
numPixels = cellfun(@numel, CC.PixelIdxList);
[~, idx] = max(numPixels);
BW2 = false(size(BW));
BW2(CC.PixelIdxList{idx}) = true;

end

% =======================================================================
% Helper: disk structuring element that works across MATLAB versions
% =======================================================================
function se = safeDiskStrel(r)
try
    se = strel('disk', r);
catch
    % Older versions sometimes require the "n" argument
    se = strel('disk', r, 0);
end
end
