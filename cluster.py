
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# RETRIEVE GRAYSCALE IMAGE 
gbr = {
    'gbr1':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb1.jpg'), cv2.COLOR_BGR2GRAY),
    'gbr2':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb2.jpg'), cv2.COLOR_BGR2GRAY),
    'gbr3':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb3.jpg'), cv2.COLOR_BGR2GRAY),
    'gbr4':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb4.jpg'), cv2.COLOR_BGR2GRAY),
    'gbr5':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb5.jpg'), cv2.COLOR_BGR2GRAY),
    'gbr7':cv2.cvtColor(cv2.imread('projects/static/img/landsat7/gb7.jpg'), cv2.COLOR_BGR2GRAY),
    
}
fitur = []
for i in range(len(gbr['gbr1'])):
    for j in range(len(gbr['gbr1'])):
        pixel = [
            gbr['gbr1'][i][j],
            gbr['gbr2'][i][j],
            gbr['gbr3'][i][j],
            gbr['gbr4'][i][j],
            gbr['gbr5'][i][j],
            gbr['gbr7'][i][j]
        ]
        fitur.append(pixel)
fitur = np.asarray(fitur)
fitur = fitur.copy()

# CLUSTERING WITH KMEANS OPENCV
fitur = np.float32(fitur)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 10.0)
K = 6
ret, label, center = cv2.kmeans(fitur,K,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)

arrclus = ['cA', 'cB', 'cC', 'cD', 'cE', 'cF']
colors = ['r', 'g', 'b', 'c', 'm', 'y']
clusters = {}
        
for i in range(0,K):
    clusters[arrclus[i]] = fitur[label.ravel() == i]

for i in range(0,K):
    plt.scatter(clusters[arrclus[i]][:,0], clusters[arrclus[i]][:,1], c=colors[i])

plt.scatter(center[:,0],center[:,1],s=80,c='k',marker='o')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.savefig('projects/static/img/landsat7/cluster-'+str(K)+'.png')
