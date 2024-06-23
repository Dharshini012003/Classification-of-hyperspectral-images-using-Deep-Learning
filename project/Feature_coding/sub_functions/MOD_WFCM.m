clc;
clear all;
close all;
 
%%%%Get the image Files from Current Directory
[f p]=uigetfile('*.*');
Inp=imread([p f]);
% Inp = imresize(Inp,[256,256]);
 

figure,imshow(Inp);
    
B=entropy(Inp)

% ldr = waitbar(0,'Pleasewait....Processing...');
dInp = Inp;
 
[rn cn] = size(dInp);      %%%%%%%%%%%%%unsupervised segmentation 
len = rn*cn;
data = double(reshape(dInp,[len 1]));
dims = [rn cn];
% Consider the parameters for clustering process
Cluster = 3;
[data_n in_n]= size(data);
   
expo = 2;       % Exponent for U
max_iter = 100;     % Max. iteration
min_impro = 1e-5;       % Min. improvement
display = 0;        % Display info or not
ncluster = 3;
nwin =5; spw =1; mfw =1;
 
obj_fcn = zeros(max_iter, 1);   % Array for objective function
% Initialize fuzzy partition
U = rand(Cluster, data_n);
col_sum = sum(U);
U = U./col_sum(ones(Cluster, 1), :);
           
% ld = waitbar(0,'PleaseWait...Processing');
for i = 1:max_iter
 
% MF matrix after exponential modification  
mf = U.^expo; 
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % new center
% fill the distance matrix
 
dist = zeros(size(center, 1), size(data, 1));
 
% fill the output matrix
 
if size(center, 2) > 1,
    for k = 1:size(center, 1),
    dist(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2)'));
    end
else    
    for k = 1:size(center, 1),
    dist(k, :) = abs(center(k)-data)';
    end
end
 
obj_fcn(i) = sum(sum((dist.^2).*mf));  % objective function
tmp = dist.^(-2/(expo-1));     
U_new = tmp./(ones(Cluster, 1)*sum(tmp));
 
tempwin=ones(nwin);
mfwin=zeros(size(U_new));
 
for j=1:size(U_new,1)
    tempmf=reshape(U_new(j,:), dims);
    tempmf=imfilter(tempmf,tempwin,'conv');
    mfwin(j,:)=reshape(tempmf,1,size(U_new,2));
end
 
mfwin=mfwin.^spw;
U_new=U_new.^mfw;
   
tmp=mfwin.*U_new;
U_new=tmp./(ones(Cluster, 1)*sum(tmp));
U = U_new;    
    
    if display, 
        fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
    end
%     check termination condition
    if i > 1,
        if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end,
    end
%     waitbar(i/max_iter,ld);
end
% close(ld);
 
iter_n = i; % Actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];
 
%figure('Name','KWFCM: Segmented Results','MenuBar','none');
for i=1:ncluster
    rimg=reshape(U(i,:,:),size(Inp,1),size(Inp,2));
    subplot(2,2,i); imshow(rimg,[])
    title(['Cluster: ' int2str(i)])
    nfile = strcat(int2str(i),'.jpg');
    imwrite(rimg,nfile)
end
 
%%%% Morphological process%%%%%%%%%%%%%
%  prompt={'Enter the number'};
% dlg_title = 'Input ';
% num_lines = 1;
% 
% answer = cell2mat(inputdlg(prompt,dlg_title,num_lines));
answer=3;
I=imread([num2str(answer) '.jpg']);
I = im2bw(I);
%figure,imshow(I)



bw=bwlabel(I);
stat=regionprops(bw,'Area');
A=[stat.Area];
for i=1:numel(A)
    if A(i)>800 | A(i)<150 
     bw(find(bw==i))=0;
    end
end
bw(find(bw>0))=1;
bw1=imfill(bw,'holes');
%figure,imshow(bw)


se=strel('disk',6);
bw1=imdilate(bw1,se);
%figure,imshow(bw1)


inn=double(Inp).*double(bw1);
figure,imshow(inn,[])
B2=entropy(inn)

%inp=imread('plasma2.bmp');
%B=entropy(inp);

ref=Inp
% ref = imread('plasma2.bmp');
H = fspecial('Gaussian',[11 11],1.5);
A = imfilter(ref,H,'replicate');
 
figure,subplot(1,2,1); imshow(ref); title('Reference Image');
subplot(1,2,2); imshow(A);   title('Blurred Image');
[ssimval, ssimmap] = ssim(A,ref);
  fprintf('The SSIM value is %0.4f.\n',ssimval);
  figure, imshow(ssimmap,[]);
title(sprintf('ssim Index Map - Mean ssim Value is %0.4f',ssimval));

H = fspecial('Gaussian',[11 11],1.5);
A = imfilter(inn,H,'replicate');
[ssimval1, ssimmap1] = ssim(inn,A);
fprintf('The SSIM value is %0.4f.\n',ssimval1);
  figure, imshow(ssimmap1,[]);
title(sprintf('ssim Index Map - Mean ssim Value is %0.4f',ssimval1));




 

        


  


 


