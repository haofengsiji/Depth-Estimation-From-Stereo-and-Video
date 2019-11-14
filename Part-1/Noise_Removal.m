clear
clc
addpath('../GCMex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% pairwise = connected edge
% class = label
% unary = data term cost
% labelcost = prior term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%read image
img = double(imread('bayes_in.jpg'));
[height, width, ~] = size(img);

SOURCE_COLOR = [0, 0, 255]; % blue = foreground
SINK_COLOR = [245, 210, 110]; % yellow = background

img_seq = reshape(img,[],3); % scan the image by column
N = size(img_seq,1);
C = 2;
class = zeros(1,N);
unary = zeros(C,N);

labelcost = [0,1;1,0];
% for each node in image
for n = 1:N
        node = img_seq(n,:);
        % data term
        unary(1,n) = L1(node,SINK_COLOR); % B cost / class 0
        unary(2,n) = L1(node,SOURCE_COLOR); % F cost / class 1
        if unary(2,n) < unary(1,n)
            class(1,n) = 1; % F < B; F:1
        end
end

tic
% construct edge
%   locate the neighbors' location
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

% draw improved images
for lam=[1:10,20:10:100]
    [labels,E,EA] = GCMex(class,single(unary),pairwise*lam,single(labelcost));
    board = repmat(reshape(SINK_COLOR,1,3),N,1);
    idx = find(labels==1);
    board(idx,:) =  repmat(reshape(SOURCE_COLOR,1,3),size(idx,1),1);
    out_img = uint8(reshape(board,height,width,3));
    if ~exist('result', 'dir')
       mkdir('result')
    end
    imwrite(out_img,sprintf('result/out_lam_%d.jpg', lam)) 
end




% L1 distance between two pixes'value
%   input: x: 1x3 y: 1x3
function dist = L1(x,y)
    temp = x - y;
    dist = mean(abs(temp),'all');
end

