function predicted  = predictions(img,col,row)
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
if R(row,col) == 0 && G(row,col) == 1 && B(row,col) == 1
    predicted = 1;
elseif R(row,col) == 0 && G(row,col) == 1 && B(row,col) == 0
    predicted = 2;
elseif R(row,col) == 1 && G(row,col) == 1 && B(row,col) == 0
    predicted = 3;
elseif R(row,col) == 0 && G(row,col) == 0 && B(row,col) == 1
    predicted = 4;
end