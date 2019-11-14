clear
clc
addpath('../GCMex')
a = 1;
b = 2;
c = 3;
% load the cameras matrix in cameras.txt
fileID = fopen('Road/cameras.txt','r');
num = fscanf(fileID,'%f',1);
for i = 1:num
K{i} = fscanf(fileID,'%f',[3,3])';
R{i} = fscanf(fileID,'%f',[3,3])';
T{i} = fscanf(fileID,'%f',[3,1]);
end
fclose(fileID);

K1 = K{a};
R1 = R{a};
t1 = T{a};
K2 = K{b};
R2 = R{b};
t2 = T{b};
K3 = K{c};
R3 = R{c};
t3 = T{c};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% pairwise = edge
% class = label
% unary = data term cost
% labelcost = prior term cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load images
img_path = './Road/src';
img_f = dir(fullfile(img_path,'*.jpg'));
files={img_f.name};
tic
for k=1:numel(files)
  imgs{k}= double(imread(fullfile(img_path,files{k})))/255;
end
toc

% read image
img1 = imgs{a};
img2 = imgs{b};
img3 = imgs{c};
[height,width, ~] = size(img1);

img_seq1 = reshape(img1,[],3); % scan the image by column
img_seq2 = reshape(img2,[],3); % scan the image by column
img_seq3 = reshape(img3,[],3); % scan the image by column
N = size(img_seq1,1);
D = linspace(0,0.01,50);
C = size(D,2);
unary = ones(C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, (X - Y).*(X - Y));
labelcost = labelcost./25;
tic

w = 124;%124 724 far close
h = 270;%270 481 
x1 = [w; h; 1];
x2 = K2*R2'*(R1/K1)*x1+K2*R2'*(t1-t2)*D;
x2_h = repmat(x2(3,:),3,1);
x2 = round(x2./x2_h);
x3 = K3*R3'*(R1/K1)*x1+K3*R3'*(t1-t3)*D;
x3_h = repmat(x3(3,:),3,1);
x3 = round(x3./x3_h);
figure()
imshow(img1);
axis on
hold on;
plot(x1(1),x1(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off

figure()
imshow(img2);
axis on
hold on;
plot(x2(1,:),x2(2,:), 'yx', 'MarkerSize', 2, 'LineWidth', 0.5); 
pl.Color = 'y';
pl.LineWidth = 2; 
hold off

figure()
imshow(img3);
axis on
hold on;
plot(x3(1,:),x3(2,:), 'yx', 'MarkerSize', 2, 'LineWidth', 0.5); 
pl.Color = 'y';
pl.LineWidth = 2; 
hold off










 