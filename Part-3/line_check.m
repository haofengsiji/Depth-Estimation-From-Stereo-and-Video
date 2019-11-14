clear
clc
addpath('../GCMex')

% camera matrix K,R,t
%   image1
K1 = [1221.2270770 0.0000000 479.5000000;	
      0.0000000	1221.2270770 269.5000000;	
      0.0000000	0.0000000 1.0000000];
R1 = [1.0000000000	0.0000000000	0.0000000000;	
      0.0000000000	1.0000000000	0.0000000000;	
      0.0000000000	0.0000000000	1.0000000000];
t1 = [0.0000000000	0.0000000000	0.0000000000]';
%   image2
K2 = [1221.2270770	0.0000000	479.5000000;
      0.0000000	1221.2270770	269.5000000;
      0.0000000	0.0000000	1.0000000];
R2 = [0.9998813487	0.0148994942	0.0039106989;	
      -0.0148907594	0.9998865876	-0.0022532664;	
      -0.0039438279	0.0021947658	0.9999898146];
t2 = [-9.9909793759	0.2451742154	0.1650832670]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% pairwise = edge
% class = label
% unary = data term cost
% labelcost = prior term cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read image
img1 = double(imread('test00.jpg'))/255;
img2 = double(imread('test09.jpg'))/255;
[height,width, ~] = size(img1);

img_seq1 = reshape(img1,[],3); % scan the image by column
img_seq2 = reshape(img2,[],3); % scan the image by column
N = size(img_seq1,1);
D = linspace(0,0.0085,50);
C = size(D,2);
unary = ones(C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, (X - Y).*(X - Y));
labelcost = labelcost./25;
tic

w = 124;%124 151 114 724 295 57  392
h = 270;%270 271 312 481 443 322 852
x1 = [w; h; 1];
x2 = K2*R2'*(R1/K1)*x1+K2*R2'*(t1-t2)*D;
x2_h = repmat(x2(3,:),3,1);
x2 = round(x2./x2_h);
figure()
imshow(img1);
axis on
hold on;
% Plot cross at row 100, column 50
plot(x1(1),x1(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off

figure()
imshow(img2);
axis on
hold on;
%pl = plot([203 693], [201 188]);
plot(x2(1,:),x2(2,:), 'yx', 'MarkerSize', 2, 'LineWidth', 0.5); 
pl.Color = 'y';
pl.LineWidth = 2; 
hold off
















 