clear
clc
addpath('../GCMex')

% load the cameras matrix in cameras.txt
fileID = fopen('Road/cameras.txt','r');
num = fscanf(fileID,'%f',1);
for i = 1:num
K{i} = fscanf(fileID,'%f',[3,3])'; % matlab scan file by column.
R{i} = fscanf(fileID,'%f',[3,3])';
T{i} = fscanf(fileID,'%f',[3,1]);
end
fclose(fileID);

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

a = 3;
b = 1;
c = 2;
d = 4;
e = 5;
K1 = K{a};
R1 = R{a};
t1 = T{a};
K2 = K{b};
R2 = R{b};
t2 = T{b};
K3 = K{c};
R3 = R{c};
t3 = T{c};
K4 = K{d};
R4 = R{d};
t4 = T{d};
K5 = K{e};
R5 = R{e};
t5 = T{e};


a = imgs{a};
b = imgs{b};
c = imgs{c};
d = imgs{d};
e = imgs{e};
[height,width, ~] = size(c);
img_seq1 = reshape(a,[],3); % scan the image by column
img_seq2 = reshape(b,[],3); % scan the image by column
img_seq3 = reshape(c,[],3); % scan the image by column
img_seq4 = reshape(d,[],3); % scan the image by column
img_seq5 = reshape(e,[],3); % scan the image by column
N = size(img_seq1,1);
D = linspace(0,0.01,50); %good 0-0.01 20
C = size(D,2);
unary = ones(4,C,N);
unary_c = ones(4,C,N);
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, (X - Y).*(X - Y));
sigma_c = 1;
sigma_d = 0.1; % good 0.1

tic
 for w = 1:width
     for h = 1:height
        n1 = (w-1)*height + h;
        node1 = reshape(a(h,w,:),1,3);
        x1 = [w; h; 1];
        x2 = K2*R2'*(R1/K1)*x1+K2*R2'*(t1-t2)*D;
        x2_h = repmat(x2(3,:),3,1);
        x2_c = x2./x2_h;
        x2 = round(x2_c);
        n2 = (x2(1,:)-1)*height + x2(2,:);
        x21 = K1*R1'*(R2/K2)*x2_c(:,n2<=518400 & n2>=1)+K1*R1'*(t2-t1)*D(:,n2<=518400 & n2>=1);
        x21_h = repmat(x21(3,:),3,1);
        x21 = x21./x21_h;
        n2_ls  = n2(n2<=518400 & n2>=1);
        node2s = img_seq2(n2_ls,:);
        match_cost = sqrt(sum((node1 - node2s).^2,2));
        match_cost_c = exp(-sum((x21-x1).^2,1)./(2*sigma_d^2));
        unary(1,n2<=518400 & n2>=1,n1) = match_cost;
        unary_c(1,n2<=518400 & n2>=1,n1) = match_cost_c;
        x3 = K3*R3'*(R1/K1)*x1+K3*R3'*(t1-t3)*D;
        x3_h = repmat(x3(3,:),3,1);
        x3_c = x3./x3_h;
        x3 = round(x3_c);
        n3 = (x3(1,:)-1)*height + x3(2,:);
        x31 = K1*R1'*(R3/K3)*x3_c(:,n3<=518400 & n3>=1)+K1*R1'*(t3-t1)*D(:,n3<=518400 & n3>=1);
        x31_h = repmat(x31(3,:),3,1);
        x31 = x31./x31_h;
        n3_ls  = n3(n3<=518400 & n3>=1);
        node3s = img_seq3(n3_ls,:);
        match_cost = sqrt(sum((node1 - node3s).^2,2));
        match_cost_c = exp(-sum((x31-x1).^2,1)./(2*sigma_d^2));
        unary(2,n3<=518400 & n3>=1,n1) = match_cost;
        unary_c(2,n3<=518400 & n3>=1,n1) = match_cost_c;
        x4 = K4*R4'*(R1/K1)*x1+K4*R4'*(t1-t4)*D;
        x4_h = repmat(x4(3,:),3,1);
        x4_c = x4./x4_h;
        x4 = round(x4_c);
        n4 = (x4(1,:)-1)*height + x4(2,:);
        x41 = K1*R1'*(R4/K4)*x4_c(:,n4<=518400 & n4>=1)+K1*R1'*(t4-t1)*D(:,n4<=518400 & n4>=1);
        x41_h = repmat(x41(3,:),3,1);
        x41 = x41./x41_h;
        n4_ls  = n4(n4<=518400 & n4>=1);
        node4s = img_seq4(n4_ls,:);
        match_cost = sqrt(sum((node1 - node4s).^2,2));
        match_cost_c = exp(-sum((x41-x1).^2,1)./(2*sigma_d^2));
        unary(3,n4<=518400 & n4>=1,n1) = match_cost;
        unary_c(3,n4<=518400 & n4>=1,n1) = match_cost_c;
        x5 = K5*R5'*(R1/K1)*x1+K5*R5'*(t1-t5)*D;
        x5_h = repmat(x5(3,:),3,1);
        x5_c = x5./x5_h;
        x5 = round(x5_c);
        n5 = (x5(1,:)-1)*height + x5(2,:);
        x51 = K1*R1'*(R5/K5)*x5_c(:,n5<=518400 & n5>=1)+K1*R1'*(t5-t1)*D(:,n5<=518400 & n5>=1);
        x51_h = repmat(x51(3,:),3,1);
        x51 = x51./x51_h;
        n5_ls  = n5(n5<=518400 & n5>=1);
        node5s = img_seq5(n5_ls,:);
        match_cost = sqrt(sum((node1 - node5s).^2,2));
        match_cost_c = exp(-sum((x51-x1).^2,1)./(2*sigma_d^2));
        unary(4,n5<=518400 & n5>=1,n1) = match_cost;
        unary_c(4,n5<=518400 & n5>=1,n1) = match_cost_c;
     end
end
toc
unary = sigma_c./(unary+sigma_c);
unary = unary.*unary_c; % bundle optimazation
unary = squeeze(sum(unary,1));
Ux = repmat(max(unary,[],1),C,1);
unary = 1 - unary./Ux;

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


% 
% tic 
% [labels,E,EA] = GCMex(class-1,single(unary),pairwise*0.009,single(labelcost),1);%good 0.009
% toc
% 
% d = labels-1;
% % normalize
% d = d/max(d);
% a = reshape(d,height,width);
% figure()
% imshow(a)

