clear
clc
addpath('GCMex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% pairwise = connected edge
% class = label
% unary = data term cost
% labelcost = prior term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read image
img1 = double(imread('im2.png'));
img2 = double(imread('im6.png'));
[height,width, ~] = size(img1);

img_seq1 = reshape(img1,[],3); % scan the image by column
img_seq2 = reshape(img2,[],3); % scan the image by column
N = size(img_seq1,1);
patch = 5;
D = (-50:50); % good range: 50,
C = size(D,2);
unary = 750*ones(C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = abs(X - Y);
labelcost = labelcost./max(labelcost);

 tic
 node1_patch = zeros(patch,1,3);
 % for each node in image
 patch_width = width-patch+1;
 for w = 1:patch_width
     for h = 1:height
         n = (w-1)*height + h;
         node1_patch = permute(img1(h,w:w+patch-1,:), [2 1 3]);
         node2s = squeeze(img2(h,:,:));
         % data term
         w_ls = D+w;
         match_w_ls = w_ls(1,find(w_ls>=1,1):find(w_ls<=patch_width,1,'last'));
         node1_patch = repmat(node1_patch,1,size(match_w_ls,2),1);
         match_cost_patch = zeros(patch,size(match_w_ls,2));
         for i = 1:patch
             match_cost_patch(i,:) = sum(abs(squeeze(node1_patch(i,:,:)) - node2s(match_w_ls+i-1,:)),2);
         end
         match_cost = mean(match_cost_patch,1);
         cover_start = find(w_ls>=1,1);
         cover_end = find(w_ls<=patch_width,1,'last');
         unary(cover_start:cover_end,n) = match_cost;
     end
 end
 toc

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

d = abs(class - fix(C/2));
% normalize
% d = d/max(d);
a = reshape(d,height,width);
figure()
imshow(a)
% 
% % tic
% % [labels,E,EA] = GCMex(class,single(unary),pairwise*10,single(labelcost),1);
% % toc
% % 
% % d = abs(labels - fix(C/2));
% % % normalize
% % d = d/max(d);
% % a = reshape(d,height,width);
% % figure()
% % imshow(a)







