                                                     clc;
clear;
close all;
warning('off','all');
%% Get input image
feature = [];
for xyz = 1:5
    close all;
[f,p] = uigetfile('database1\*.tiff');
[~,filename] = fileparts(f);
% read input image
I1 = imread([p,f]);
% I1 = imcrop(I);
figure,imshow(I1);
title('Input High Resolution Image');
run('vlfeat-0.9.20\toolbox\vl_root.m');
run('vlfeat-0.9.20\toolbox\vl_setup.m');
%% i) K-means
% converting to LAB color space
cform = makecform('srgb2lab');
lab_he = applycform(I1,cform);
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
imshow(pixel_labels1,[]), title('image labeled by k-medioidcluster index');

initSigmas = zeros(dimension,numClusters);
initWeights = zeros(1,numClusters); 
for i=1:numClusters
    Xk = ab(:,assignments==i);

    initWeights(i) = size(Xk,2) / numClusters;

%     plot(Xk(1,:),Xk(2,:),'.','color',cc(i,:));
    if size(Xk,1) == 0 || size(Xk,2) == 0
        initSigmas(:,i) = diag(cov(ab'));
    else
        initSigmas(:,i) = diag(cov(Xk'));
    end
end
[means,sigmas,weights,ll,posteriors] = vl_gmm(ab, numClusters, ...
                                              'Initialization','custom', ...
                                              'InitMeans',initMeans, ...
                                              'InitCovariances',initSigmas, ...
                                              'InitPriors',initWeights, ...
                                              'Verbose', ...
                                              'MaxNumIterations', 100) ;
segmented_images = cell(1,5);
rgb_label = repmat(pixel_labels1,[1 1 3]);
for k = 1:numClusters
    color = I1;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
    figure,
    imshow(segmented_images{k});
end
feature = [feature;posteriors];
pause
end
final_feature = feature;
water = [final_feature(2,:);final_feature(5,:);final_feature(9,:);final_feature(17,:)];

agriculture = [final_feature(1,:);final_feature(7,:);final_feature(11,:);final_feature(15,:);...
    final_feature(19,:)];

barreland = [final_feature(4,:);final_feature(6,:);final_feature(12,:);final_feature(13,:);...
    final_feature(16,:);final_feature(20,:)];

greenland = [final_feature(3,:);final_feature(8,:);final_feature(10,:);final_feature(14,:);...
    final_feature(18,:)];
% 
extracted_feature = [water;agriculture;barreland;greenland];
extracted_feature(:,end+1) = [1;1;1;1;...
                              2;2;2;2;2;...
                              3;3;3;3;3;3;...
                              4;4;4;4;4];
                          
save('Extracted_Feature','extracted_feature');