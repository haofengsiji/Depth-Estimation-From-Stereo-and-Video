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

obj = 3;
frames = 5;
[height,width, ~] = size(imgs{1});
N = height*width;
D = linspace(0,0.01,50); %good 0-0.01 50
C = size(D,2);
unary = ones(frames-1,C,N); % p_c
unary_c = ones(frames-1,C,N); % p_v
[X, Y] = meshgrid(1:C, 1:C);
labelcost = min(25, abs(X - Y));
sigma_c = 1;
sigma_d = 0.1; 

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
% too smooth so I cut the diagonal edge
% i = [loc_right;loc_left;loc_top;loc_bottom];
% j = [loc_left;loc_right;loc_bottom;loc_top];
pairwise = sparse(i,j,1);
toc
%:num-frames+1
for t = 8:num-frames+1
    tic
    f_out(t,frames,imgs,K,R,T,pairwise,unary,unary_c,sigma_c,sigma_d,labelcost,height,width,obj,D,C);
    toc
    clear f_out
end

function f_out(t,frames,imgs,K,R,T,pairwise,unary,unary_c,sigma_c,sigma_d,labelcost,height,width,obj,D,C)

        
        % task initialization
        for i = 1:frames
            img{i} = imgs{i+t-1};
            img_seq{i} = reshape(img{i},[],3); % scan the image by column
            K_inloop{i} = K{i+t-1};
            R_inloop{i} = R{i+t-1};
            T_inloop{i} = T{i+t-1};
        end
        for w = 1:width
            for h = 1:height
                n1 = (w-1)*height + h;
                node1 = reshape(img{obj}(h,w,:),1,3);
                x1 = [w; h; 1];
                for i = 1:frames                
                    if i == obj
                        continue
                    end
                    x2 = K_inloop{i}*R_inloop{i}'*(R_inloop{obj}/K_inloop{obj})*x1+K_inloop{i}*R_inloop{i}'*(T_inloop{obj}-T_inloop{i})*D;
                    x2_h = repmat(x2(3,:),3,1);
                    x2_c = x2./x2_h;
                    x2 = round(x2_c);
                    n2 = (x2(1,:)-1)*height + x2(2,:);
                    x21 = K_inloop{obj}*R_inloop{obj}'*(R_inloop{i}/K_inloop{i})*x2_c(:,n2<=518400 & n2>=1)+K_inloop{obj}*R_inloop{obj}'*(T_inloop{i}-T_inloop{obj})*D(:,n2<=518400 & n2>=1);
                    x21_h = repmat(x21(3,:),3,1);
                    x21 = x21./x21_h;
                    n2_ls  = n2(n2<=518400 & n2>=1);
                    node2s = img_seq{i}(n2_ls,:);
                    match_cost = sqrt(sum((node1 - node2s).^2,2));
                    match_cost_c = exp(-sum((x21-x1).^2,1)./(2*sigma_d^2));
                    if i < obj
                        unary(i,n2<=518400 & n2>=1,n1) = match_cost;
                        unary_c(i,n2<=518400 & n2>=1,n1) = match_cost_c;
                    else
                        unary(i-1,n2<=518400 & n2>=1,n1) = match_cost;
                        unary_c(i-1,n2<=518400 & n2>=1,n1) = match_cost_c;
                    end
                end
            end
        end
        unary = sigma_c./(unary+sigma_c);
        unary = unary.*unary_c; % bundle optimazation
        unary = squeeze(sum(unary,1));
        Ux = repmat(max(unary,[],1),C,1);
        unary = 1 - unary./Ux;

        % class should start from 0
        [~,class] = min(unary,[],1);

        [labels,E,EA] = GCMex(class-1,single(unary),pairwise*0.009,single(labelcost),1);%good 0.009
        d = labels-1;
        % normalize
        d = d/max(d);
        a = reshape(d,height,width);
        d0 = class-1;
        % normalize
        d0 = d0/max(d0);
        a0 = reshape(d0,height,width);
        if ~exist('result_2', 'dir')
           mkdir('result_2')
        end
        imwrite(a,sprintf('result_2/road_%d_%1.3f.jpg', obj+t-1,sigma_d)) 
        imwrite(a0,sprintf('result_2/init_road_%d_%1.3f.jpg', obj+t-1,sigma_d)) 
    end
         
            


