function [Dir,res] = PCA(Data)
% 以[-pi/2,pi/2]之间的角度pi表示方向
% 1.去除均值
[buf K] = size(Data);
miu = mean(Data')';
for k=1:K
    Data(:,k) = Data(:,k)-miu; 
end;
sigma = zeros(2,2);
% 2.计算协方差
for k=1:K
    x = Data(:,k);
    sigma = sigma+x*x';
end;
sigma = sigma/K;  
% 3.特征分解
[V,D] = eig(sigma);
if (D(1,1)<D(2,2)) % 把较大的特征值对应的向量挪到第一行
     buf = V(1,:);
     V(1,:) = V(2,:);
     V(2,:) = buf;
end;
% 4.求解主方向向量
res = inv(V)*[1;0];
Dir = atan(res(2)/res(1))/pi*180;