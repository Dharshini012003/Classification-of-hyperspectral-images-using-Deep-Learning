clc;
clear;
close all;
warning('off','all');
addpath('sub_functions\');
%% Get input image
[f,p] = uigetfile('database1\*.tiff');
[~,filename] = fileparts(f);
% read input image
I1 = imread([p,f]);
% I1 = imcrop(I);
figure,imshow(I1);
title('Input High Resolution Image');

% run('vlfeat-0.9.20\toolbox\vl_root.m');
% run('vlfeat-0.9.20\toolbox\vl_setup.m');
%% ..
% converting to LAB color space
cform = makecform('srgb2lab');
lab_he = applycform(I1,cform);
%figure,imshow(lab_he); title('L A B Image');
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
% Reshape
ab = reshape(ab,nrows*ncols,2);
ab = ab';
x = ab;
net = selforgmap([2 2]);
net = train(net,x);
view(net)
y = net(x);
classes = vec2ind(y);
pixel_labels1 = reshape(classes,512,512);

figure,
imshow(pixel_labels1,[]), title('image labeled by cluster index');

segmented_images = cell(1,4);
rgb_label = repmat(pixel_labels1,[1 1 3]);
load Extracted_Feature;
%figure,imshow(I1);
hold on
I2 = im2double(I1);
R1 = I2(:,:,1);
G1 = I2(:,:,2);
B1 = I2(:,:,3);

figure
subplot(221)
imshow(I2)
title('Original image')
subplot(222)
imshow(R1); title('Red Channel')
subplot(223)
imshow(G1); title('Green Channel')
subplot(224)
imshow(B1); title('Blue Channel')


for k = 1:4
    color = I1;
    bw = ones(size(color));
    color(rgb_label ~= k) = 0;
    bw(rgb_label ~= k) = 0;
%     figure,imshow(bw);
    segmented_images{k} = color;
%     figure,
%     imshow(segmented_images{k});

%% Color Features
I = im2double(segmented_images{k});
lab = rgb2lab(I);
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
l = lab(:,:,1);
a = lab(:,:,2);
b = lab(:,:,3);

% Mean
R_mean = mean2(R);
G_mean = mean2(G);
B_mean = mean2(B);
l_mean = mean2(l);
a_mean = mean2(a);
b_mean = mean2(b);

MEAN = [R_mean G_mean B_mean l_mean a_mean b_mean];

% Standard Deviation
R_SD = std2(R);
G_SD = std2(G);
B_SD = std2(B);
l_SD = std2(l);
a_SD = std2(a);
b_SD = std2(b);

SD = [R_SD G_SD B_SD l_SD a_SD b_SD];

% Kurtosis
R_kurtosis = kurtosis(kurtosis(R));
G_kurtosis = kurtosis(kurtosis(G));
B_kurtosis = kurtosis(kurtosis(B));
l_kurtosis = kurtosis(kurtosis(l));
a_kurtosis = kurtosis(kurtosis(a));
b_kurtosis = kurtosis(kurtosis(b));

KURTOSIS = [R_kurtosis G_kurtosis B_kurtosis l_kurtosis a_kurtosis b_kurtosis];

% Skewness
R_skewness = skewness(skewness(R));
G_skewness = skewness(skewness(G));
B_skewness = skewness(skewness(B));
l_skewness = skewness(skewness(l));
a_skewness = skewness(skewness(a));
b_skewness = skewness(skewness(b));

SKEWNESS = [R_skewness G_skewness B_skewness l_skewness a_skewness b_skewness];

feature = [MEAN SD KURTOSIS SKEWNESS];

end