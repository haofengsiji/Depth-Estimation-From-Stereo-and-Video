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
D = linspace(0,0.0085,50); %good 0-0.085 50
C = size(D,2);
unary = ones(C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, (X - Y).*(X - Y));
tic
 for w = 1:width
     for h = 1:height
        n1 = (w-1)*height + h;
        node1 = reshape(img1(h,w,:),1,3);
        x1 = [w; h; 1];
        x2 = K2*R2'*(R1/K1)*x1+K2*R2'*(t1-t2)*D;
        x2_h = repmat(x2(3,:),3,1);
        x2 = round(x2./x2_h);  
        n2 = (x2(1,:)-1)*height + x2(2,:);
        n2_ls  = n2(n2<=518400 & n2>=1);
        node2s = img_seq2(n2_ls,:);
        match_cost = sqrt(sum((node1 - node2s).^2,2));
        unary(n2<=518400 & n2>=1,n1) = match_cost;
     end
end
toc

% class should start from 0
[~,class] = min(unary,[],1);

% construct edge
%   locate the neighbors' location
tic
loc_board = ones(height,width);
loc_right = find(imtranslate(loc_board,[-1, 0],'FillValues',0) ~= 0);
loc_left = find(imtranslate(loc_board,[1, 0],'FillValues',0) ~= 0);
loc_top = find(imtranslate(loc_board,[0,1],'FillValues',0) ~= 0);
loc_bottom = find(imtranslate(loc_board,[0,-1],'FillValues',0) ~= 0);
loc_rt = find(imtranslate(loc_board,[-1, 1],'FillValues',0) ~= 0);
loc_lt = find(imtranslate(loc_board,[1, 1],'FillValues',0) ~= 0);
loc_rb = find(imtranslate(loc_board,[-1, -1],'FillValues',0) ~= 0);
loc_lb = find(imtranslate(loc_board,[1, -1],'FillValues',0) ~=0);
% i = [loc_right;loc_left;loc_top;loc_bottom;loc_rt;loc_lt;loc_rb;loc_lb];
% j = [loc_left;loc_right;loc_bottom;loc_top;loc_lb;loc_rb;loc_lt;loc_rt];
% too smooth so I cut the diagonal edge
i = [loc_right;loc_left;loc_top;loc_bottom];
j = [loc_left;loc_right;loc_bottom;loc_top];
pairwise = sparse(i,j,1);
toc

d = class-1;
% normalize
d = d/max(d);
a = reshape(d,height,width);
figure()
imshow(a)


tic 
[labels,E,EA] = GCMex(class-1,single(unary),pairwise*0.015,single(labelcost),1);%best 0.015
toc

d = labels-1;
% normalize
d = d/max(d);
a = reshape(d,height,width);
figure()
imshow(a)
















 