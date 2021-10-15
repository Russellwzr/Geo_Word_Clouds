clc,clear
[A,B] = xlsread('CityTags2.xlsx');
[Geo,GeoName] = xlsread('Region.xlsx');
Geo = Geo(:,1:4);
num = length(A);
pre = B(1);
x = [];
y = [];
idx = [];
C = [];
D = [];
Region = [];
Tags = [];
Range = [];
% 方向：水平 = 1，竖直 = 0
Dir = [];
% 第 cnt 个聚类
cnt = 1;
% 为 输入信息 打上聚类标签
cluster_points = [];
for i = 1:num
    cur = B(i);
    if ~isequal(pre,cur)
        X = [x' y'];
        Region = unique(Region);
        Rsz = size(Region);
        Region = [];
        K = Rsz(2);
        [idx,D] = kmeans(X,K);
        for cluster = 1:K
            CurCluster = [X(idx==cluster,1) X(idx==cluster,2)];
            CurCluster_Size = size(CurCluster);
            tmp = zeros(CurCluster_Size(1),1);
            tmp = tmp + cnt;
            cluster_points = [cluster_points;X(idx==cluster,1) X(idx==cluster,2) tmp];
            direction = abs(PCA_Rotation(CurCluster'));
            if direction > 45
                Dir = [Dir;0];
            else
                Dir = [Dir;1];
            end
            sz = sum(idx==cluster);
            tmpX = sort(X(idx==cluster,:));
            min_x = tmpX(ceil(0.05*sz),1);
            max_x = tmpX(ceil(0.95*sz),1);
            min_y = tmpX(ceil(0.05*sz),2);
            max_y = tmpX(ceil(0.95*sz),2);
            Range = [Range;min_x,max_x,min_y,max_y];
            C = [C;D(cluster,1) D(cluster,2) sz];
            cnt = cnt + 1;
            x = [];
            y = [];
            Tags = [Tags;string(pre)];
        end
    end
    x = [x A(i,1)];
    y = [y A(i,2)];
    Region = [Region A(i,3)];
    pre=cur;
end
X = [x' y'];
Region = unique(Region);
Rsz = size(Region);
K = Rsz(2);
[idx,D] = kmeans(X,K);
for cluster = 1:K
    CurCluster = [X(idx==cluster,1) X(idx==cluster,2)];
    CurCluster_Size = size(CurCluster);
    tmp = zeros(CurCluster_Size(1),1);
    tmp = tmp + cnt;
    cluster_points = [cluster_points;X(idx==cluster,1) X(idx==cluster,2) tmp];
    direction = abs(PCA_Rotation(CurCluster'));
    if direction > 45
        Dir = [Dir;0];
    else
        Dir = [Dir;1];
    end
    sz = sum(idx==cluster);
    Range = [Range;min(X(idx==cluster,1)),max(X(idx==cluster,1)),min(X(idx==cluster,2)),max(X(idx==cluster,2))];
    C = [C;D(cluster,1) D(cluster,2) sz];
    cnt = cnt + 1;
    x = [];
    y = [];
    Tags = [Tags;string(cur)];
end
xlswrite('CityTagsCluster2.xlsx', [Tags,C,Dir,Range],1, 'A2')
xlswrite('CityTagsWithIndex2.xlsx',cluster_points,1,'A2')
