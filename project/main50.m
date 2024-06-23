clc;
clear;
close all;
warning('off','all');
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
%% i) Self Organising MAP based clustering
% converting to LAB color space
cform = makecform('srgb2lab');
lab_he = applycform(I1,cform);
figure,imshow(lab_he); title('L A B Image');
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
% Reshape_
ab = reshape(ab,nrows*ncols,2);
ab = ab';
x = ab;
net = selforgmap([2 2]);
%net.trainParam.epochs = 500;
net = train(net,x);
view(net)
y = net(x);
classes = vec2ind(y);
pixel_labels1 = reshape(classes,512,512);

%---------//to show input for som//---------run in cmd
%neuron_1_input_indices = find(classes ==1)  //indices or vectors values
%neuron_1_input_values = x(neuron_1_input_indices) //indices to exact values
%---------//to show input for som//---------run in cmd
figure,
imshow(pixel_labels1,[]), title('image labeled by cluster index');


% //Generate colored labels 
colored_labels = label2rgb(pixel_labels1, 'hsv', 'k', 'shuffle');

% //Overlay the colored labels on the image
hold on;
h = imshow(colored_labels);
set(h, 'AlphaData', 0.5); % //Set transparency to make the underlying image visible

%// Add labels for interpretation outside the image on the rightmost side
text(size(pixel_labels1, 2) + 30, 20, 'Agriculture', 'Color', 'Green', 'FontSize', 12, 'HorizontalAlignment', 'left'); % Adjust position
rectangle('Position', [size(pixel_labels1, 2) + 5, 10, 10, 10], 'FaceColor', 'g'); % Adjust position

text(size(pixel_labels1, 2) + 30, 40, 'Water', 'Color', 'Cyan', 'FontSize', 12, 'HorizontalAlignment', 'left'); % Adjust position
rectangle('Position', [size(pixel_labels1, 2) + 5, 30, 10, 10], 'FaceColor', 'b'); % Adjust position
hold off;

%-----------------------------------------------------------------------------------

segmented_images = cell(1,4);
rgb_label = repmat(pixel_labels1,[1 1 3]);