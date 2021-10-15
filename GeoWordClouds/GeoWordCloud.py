from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import time

fig,ax=plt.subplots()

'''
# Load Geo Map
dataf = pd.read_excel('GeoRegion.xlsx')
wordmap = dataf.values
sz = wordmap.shape
geomap = dict()
for i in range(0,sz[0]):
    geomap[wordmap[i][0]] =[wordmap[i][1],wordmap[i][2],wordmap[i][3],wordmap[i][4]]
'''

# Load Data Set
# Load Cluster Info
df = pd.read_excel('CityTagsCluster1.xlsx')
words = df.values
# Load Tags labeled by cluster index
df2 = pd.read_excel('CityTagsWithIndex1.xlsx')
points = df2.values
cluster_size = words.shape[0] + 10
cluster_points = [[] for i in range(cluster_size)]

for i in range(0, points.shape[0]):
    cluster_points[points[i][2]].append((points[i][0],points[i][1]))

# compute the area of M in pixel : Am
img = cv2.imread("SD.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
h, w = gray.shape[:2]
m = np.reshape(gray, [1, w * h])
mean = m.sum() / (w * h)
ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
Am = len(binary[binary == 0])

# the area of the rectangle bounding region M
A = 680 * 440

# set parameters
shape=np.array(Image.open(r'SD.png'))
stopwords = set(STOPWORDS)

starttime = time.time()

# draw the geo word cloud
mycloudword=WordCloud(font_path=r'C:\Windows\Fonts\msyh.ttc',
                      width=800,
                      height=500,
                      contour_width=1,
                      contour_color='steelblue',
                      background_color='white',
                      mask=shape,
                      font_step=1,
                      stopwords=stopwords,
                      collocations=False,
                      random_state=50).GeoGenerate(words, A, Am,cluster_points, points.shape[0])

ax.imshow(mycloudword)
ax.axis("off")
plt.show()

mycloudword.to_file("output.png")

endtime = time.time()
dtime = endtime - starttime

print("程序运行时间：%.8s s" % dtime)
