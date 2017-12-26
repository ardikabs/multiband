import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class Multiband:

    def __init__(self, path):
        self.path = path

    
    def imread(self):
        # self.path = projects/static/img/landsat7/
        gbr = {
            'gbr1':cv2.imread(self.path+"gb1.jpg",0),
            'gbr2':cv2.imread(self.path+"gb2.jpg",0),
            'gbr3':cv2.imread(self.path+"gb3.jpg",0),
            'gbr4':cv2.imread(self.path+"gb4.jpg",0),
            'gbr5':cv2.imread(self.path+"gb5.jpg",0),
            'gbr7':cv2.imread(self.path+"gb7.jpg",0),
        }
        return gbr
    
    def combine(self,image):
        feature = []
        for i in range(len(image['gbr1'])):
            for j in range(len(image['gbr1'])):
                pixel = [
                    image['gbr1'][i][j],
                    image['gbr2'][i][j],
                    image['gbr3'][i][j],
                    image['gbr4'][i][j],
                    image['gbr5'][i][j],
                    image['gbr7'][i][j],
                ]
                feature.append(pixel)
        feature = np.asarray(feature)
        return feature

    def kmeans(self, feature, cluster, iter):
        feature = feature.copy()
        feature = np.float32(feature)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iter, 10.0)
        ret, label, center = cv2.kmeans(feature, cluster, None, criteria, iter, cv2.KMEANS_RANDOM_CENTERS)
        return label, center
    
    def linkage(self,feature):
        feature = feature.copy()
        models = AgglomerativeClustering(n_clusters=6,linkage="complete")
        models.fit(feature)
        labels = models.labels_
        return labels

    def imcreate(self,  feature, label, cluster, identifier):
        img = [[0,0,0]] * len(feature)
        feature_list = feature.tolist()
        colors = {
            "3":[
                [189, 195, 199], # grey
                [245, 176, 65], # green
                [51, 79, 255]], # blue
            "4" :[
                [231, 76, 60], # red
                [51, 79, 255], # blue
                [245, 176, 65], # green
                [189, 195, 199]], # grey
            "5" :[
                [231, 76, 60], # red
                [51, 79, 255], # blue
                [245, 176, 65], # green
                [233, 255, 51], # yellow
                [189, 195, 199]], # grey
            "6" :[
                [231, 76, 60], # red
                [51, 79, 255], # blue
                [245, 176, 65], # green
                [233, 255, 51], # yellow
                [240, 128, 128], # brown
                [189, 195, 199]], # grey
        }

        for i in range(cluster):
            for j in feature[label.ravel()==i]:
                j = j.tolist()
                index = feature_list.index(j)
                img[index] = colors[str(cluster)][i]
        
        img_creation = []
        num_pixel = -1
        for i in range(0, 32):
            row = []
            for j in range(0, 32):
                num_pixel = num_pixel + 1
                piksel = img[num_pixel]
                row.append(piksel)
            img_creation.append(row)
        img_creation = np.asarray(img_creation, dtype=np.uint8)
        cv2.imwrite(self.path+"result-cluster-"+ identifier +"-"+ str(cluster) + '.jpg', img_creation)

    def chcreate(self, feature, label, cluster, identifier):
        # CREATE CLUSTER CHART
        arrclus = ['cA', 'cB', 'cC', 'cD', 'cE', 'cF']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        clusters = {}

        for i in range(0, cluster):
            clusters[arrclus[i]] = feature[label.ravel() == i]

        for i in range(0, cluster):
            plt.scatter(clusters[arrclus[i]][:,0], clusters[arrclus[i]][:,1], c=colors[i])

        # plt.scatter(center[:,0], center[:,1], s = 80, c = 'k', marker = 's')
        plt.xlabel('Height'),plt.ylabel('Weight')
        plt.savefig(self.path+"chart-" + identifier + "-" + str(cluster) + '.png')

        



