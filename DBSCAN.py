import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

class DBSCAN(object):

    def __init__(self,x,epsilon,minpts):


        # The number of points
        self.n=x.shape[0]

        # Euclidean distance
        self.p=copy.deepcopy(x)
        self.q=copy.deepcopy(x)

        # Label as visited points and noise
        self.visited=np.full((self.n),False)
        self.noise=np.full((self.n),False)

        #DBSCAN Parameters
        self.epsilon=epsilon
        self.minpts=minpts

        #Cluster
        self.idx=np.full((self.n),0)
        # The index of Cluster
        self.C=0
        self.input=x



    def regionQuery(self,i):
        self.dist = np.sqrt(np.sum(((self.p[:,np.newaxis] - self.q[i])**2),2))
        g=self.dist[i,:]<self.epsilon
        Neighbors=np.where(g==True)[0].tolist()

        return Neighbors

    def expandCluster(self,i):

            self.idx[i]=self.C
            k=0

            while True:
                try:
                    j=self.neighbors[k]
                except:pass
                if self.visited[j]!=True:
                    self.visited[j]=True

                    self.neighbors2=self.regionQuery(j)

                    if len(self.neighbors2)>self.minpts:
                        self.neighbors=self.neighbors+self.neighbors2
                
                if self.idx[j]==0:
                    self.idx[j]=self.C

                k+=1
                if len(self.neighbors)<k:
                    return

    def run(self):
        #Clustering
        for i in tqdm(range(self.n)):
            if self.visited[i]==False:
                self.visited[i]=True
                self.neighbors=self.regionQuery(i)
                if len(self.neighbors)>self.minpts:
                    self.C+=1
                    self.expandCluster(i)

                else:
                    self.noise[i]=True

        return self.idx,self.noise

    