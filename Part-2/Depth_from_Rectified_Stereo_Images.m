clear
clc
addpath('../GCMex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% pairwise = edge
% class = label
% unary = data term cost
% labelcost = prior term cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read image
img1 = double(imread('im2.png'));
img2 = double(imread('im6.png'));
[height,width, ~] = size(img1);

img_seq1 = reshape(img1,[],3); % scan the image by column
img_seq2 = reshape(img2,[],3); % scan the image by column
N = size(img_seq1,1);
D = flip([-50:0]); % good range: 50,
C = size(D,2);
unary = 4000*ones(C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, (X - Y).*(X - Y));
labelcost = labelcost./25;
tic
 % for each node in image
 for w = 1:width
     for h = 1:height
         n = (w-1)*height + h;
         node1 = reshape(img1(h,w,:),1,3);
         node2s = squeeze(img2(h,:,:));
         % data term
         w_ls = D+w;
         match_w_ls = w_ls(1,find(w_ls<=450,1):find(w_ls>=1,1,'last'));
         match_cost = sum((node1 - node2s(match_w_ls,:)).^2,2);
         cover_start = find(w_ls<=450,1);
         cover_end = find(w_ls>=1,1,'last');
         unary(cover_start:cover_end,n) = match_cost;
     end
 end
 toc
% class should start from 0, for now, class start from 1!
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
i = [loc_right;loc_left;loc_top;loc_bottom;loc_rt;loc_lt;loc_rb;loc_lb];
j = [loc_left;loc_right;loc_bottom;loc_top;loc_lb;loc_rb;loc_lt;loc_rt];
pairwise = sparse(i,j,1);
toc

d = class-1;
% normalize
d = d/max(d);
a = reshape(d,height,width);
figure()
imshow(a)

tic
unary = unary./4000;
toc
tic
[labels,E,EA] = GCMex(class-1,single(unary),pairwise*0.5,single(labelcost),1);
toc

d = labels;
% normalize
d = d/max(d);
a = reshape(d,height,width);
figure()
imshow(a)







