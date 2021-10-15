clc,clear
[A,B] = xlsread('CityTags1.xlsx');
num = length(A);
f1 = figure(1)
img = imread('SD.png');
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
for i = 1:num
    cur = B(i);
    if ~isequal(pre,cur)
        scatter(x,y,40,c(from:to),'filled')
        hold on;
        x = [];
        y = [];
        from = from + 3;
        to = to + 3;
    end
    x = [x A(i,1)];
    y = [y A(i,2)];
    pre=cur;
end
from = from + 3;
to = to + 3;
scatter(x,y,40,c(from:to),'filled')