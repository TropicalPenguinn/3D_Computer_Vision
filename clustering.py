from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
#from DBSCAN import DBSCAN


"""
pcd = o3d.io.read_point_cloud("/home/airlab/catkin_workspace/src/realsense2_description/3D_Object_Reconstruction/result{}.pcd".format(5))

points=np.asarray(pcd.points)

dbscan=DBSCAN(points,0.01,150)
dbscan.run()

"""

for i in range(1,5):
    pcd = o3d.io.read_point_cloud("/home/airlab/catkin_workspace/src/realsense2_description/3D_Object_Reconstruction/result{}.pcd".format(i))
    points=np.asarray(pcd.points)

    clustering=DBSCAN(eps=0.01,min_samples=150).fit(points)

    label=clustering.labels_
    label_0=np.asarray(np.where(label==0)[0])
    label_1=np.asarray(np.where(label==1)[0])


    pcd_0=pcd.select_by_index(label_0)
    pcd_1=pcd.select_by_index(label_1)

    o3d.visualization.draw_geometries([pcd_0])
    o3d.visualization.draw_geometries([pcd_1])



