clc,clear
[A,B] = xlsread('CityTags1.xlsx');
num = length(A);
f1 = figure(1)
img = imread('output1.png');
imagesc([0,800],[500,0],flipdim(img,1));
hold on;
xlim([0,800])
ylim([0,100])
xticks(0:100:800)
yticks(0:100:500)
pre = B(1);
x = [];
y = [];
c = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],[0,0,0],[0,0.4470,0.7410],[0.8500,0.3250,0.0980],[0.9290,0.6940,0.1250],[0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880],[0.3010,0.7450,0.9330],[0.6350,0.0780,0.1840],[0.5,0.5,0.5],[0.8,0.8,0.8],[0.2,0.2,0.2],[0.6,0.6,0.6]];
from =1;
to = 3;
idx = [];
C = [];
D = [];
Tags = [string(B(1))];
% Ë®Æ½ = 1£¬ÊúÖ± = 0
Dir = [];
for i = 1:num
    cur = B(i);
    if ~isequal(pre,cur)
        scatter(x,y,40,c(from:to),'filled')
        X = [x' y'];
        [idx,D] = kmeans(X,1);
        direction = abs(PCA_Rotation(X'))
        if direction > 45
            Dir = [Dir;0];
        else
            Dir = [Dir;1];
        end
        sz = size(idx);
        C = [C;D sz(1)];
        plot(D(1,1),D(1,2),'kx','MarkerSize',20,'LineWidth',5)
        hold on;
        x = [];
        y = [];
        from = from + 3;
        to = to + 3;
        Tags = [Tags;string(cur)];
    end
    x = [x A(i,1)];
    y = [y A(i,2)];
    pre=cur;
end
from = from + 3;
to = to + 3;
X = [x' y'];
direction = abs(PCA_Rotation(X'))
if direction > 45
    Dir = [Dir;0];
else
    Dir = [Dir;1];
end
[idx,D] = kmeans(X,1);
sz = size(idx);
C = [C;D sz(1)];
plot(D(1,1),D(1,2),'kx','MarkerSize',20,'LineWidth',5)
hold on;
scatter(x,y,40,c(from:to),'filled')
%xlswrite('CityTagsCluster.xlsx', [Tags,C,Dir], 'A1')

