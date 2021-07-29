import pandas as panda
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import matplotlib.cm as cm
import time
import matplotlib
from collections import OrderedDict
import math
from csv import writer
from csv import reader

from PIL import ImageTk, Image
import os
current_path = os.path.dirname(os.path.abspath(__file__))

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import ctypes

"""
Known issues:
1. K higher than 4 takes time to initialize : This is due to the implmented customized centroids initialization | Remarks: Perhaps use Kmeans++ or naive sharding to initialize centroids
2. Max K currently is 10 due to the initialized cmap and markers for plotting purposes
3. Centroid star colour hasn't been fixed
4. Centroids update not correct
"""

class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(500, 300)
        self.setWindowTitle('K-means [Woven Planet]')
        self.center()
        
        #Create Dropdownlist
        self.ddl = QtWidgets.QComboBox(self)
        self.ddl.addItem("L2 Norm")
        self.ddl.addItem("L1 Norm")
        self.ddl.move(160, 60)
        self.ddl.resize(100,30)
        self.ddl.setFont(QFont('century', 12))
        #self.ddl.activated[str].connect(self.onChanged_ddl(self, ))
        
        # Create textbox (cluster)
        self.textbox_cluster = QLineEdit(self)
        self.textbox_cluster.move(160, 20)
        self.textbox_cluster.resize(100,30)
        self.textbox_cluster.setFont(QFont('century', 12))
        
        # Create textbox (numData)
        self.textbox_numData = QLineEdit(self)
        self.textbox_numData.move(160, 180)
        self.textbox_numData.resize(100,30)
        self.textbox_numData.setFont(QFont('century', 12))
        
        # Create textbox (train dataset)
        self.textbox_trainDataset = QLineEdit(self)
        self.textbox_trainDataset.move(160, 100)
        self.textbox_trainDataset.resize(220,30)
        self.textbox_trainDataset.setFont(QFont('century', 10))
        
        # Create textbox (test dataset)
        self.textbox_testDataset = QLineEdit(self)
        self.textbox_testDataset.move(160, 140)
        self.textbox_testDataset.resize(220,30)
        self.textbox_testDataset.setFont(QFont('century', 10))
        
        # Create label (Enter number of clusters(K):)
        self.label_clusterA = QLabel('Enter number', self)
        self.label_clusterA.move(40, 10)
        self.label_clusterA.setFont(QFont('century', 12))
        self.label_clusterB = QLabel('of clusters (K):', self)
        self.label_clusterB.move(42, 30)
        self.label_clusterB.setFont(QFont('century', 12))
        
        # Create label (Norm)
        self.label_author = QLabel('Norm:', self)
        self.label_author.move(40, 65)
        self.label_author.setFont(QFont('century', 12))
        
        # Create label (Number of data)
        self.label_author = QLabel('Number of data:', self)
        self.label_author.move(40, 185)
        self.label_author.setFont(QFont('century', 12))
        
        # Create label (Maximum of 100 is allowed)
        self.label_clusterA = QLabel('Maximum of 100 is allowed', self)
        self.label_clusterA.move(265, 185)
        self.label_clusterA.setFont(QFont('century', 12))
        
        # Create label (Train Dataset)
        self.label_author = QLabel('Train Dataset:', self)
        self.label_author.move(40, 105)
        self.label_author.setFont(QFont('century', 12))
        
        # Create label (Test Dataset)
        self.label_author = QLabel('Test Dataset:', self)
        self.label_author.move(40, 145)
        self.label_author.setFont(QFont('century', 12))
        
        # Create label (Author: Fairuz Safwan)
        self.label_author = QLabel('Author: Fairuz Safwan', self)
        self.label_author.move(320, 270)
        self.label_author.setFont(QFont('century', 12))
        
        # Create a button in the window (Compute)
        self.button_compute = QPushButton('Compute', self)
        self.button_compute.move(200,240)
        self.button_compute.resize(80,30)
        self.button_compute.setFont(QFont('century', 12))
        
        # Create a button in the window (Compute)
        self.button_browse_trainDataset = QPushButton('Browse', self)
        self.button_browse_trainDataset.move(385,100)
        self.button_browse_trainDataset.resize(80,30)
        self.button_browse_trainDataset.setFont(QFont('century', 12))
        
        # Create a button in the window (Compute)
        self.button_browse_testDataset = QPushButton('Browse', self)
        self.button_browse_testDataset.move(385,140)
        self.button_browse_testDataset.resize(80,30)
        self.button_browse_testDataset.setFont(QFont('century', 12))
        
        # connect button to function on_click
        self.button_compute.clicked.connect(self.on_click_compute)
        self.button_browse_trainDataset.clicked.connect(self.on_click_browse_trainDataset)
        self.button_browse_testDataset.clicked.connect(self.on_click_browse_testDataset)
        
        self.show()
    
    def onChanged_ddl(self, text):
        #self.ddl.setText(text)
        self.ddl.setCurrentText(text)
        self.ddl.setFont(QFont('century', 12))
    
    
    def on_click_browse_trainDataset(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', ".\\" , '*.csv') #QtCore.QDir.rootPath()
        #Assign path to Train Dataset textbox
        self.textbox_trainDataset.setText(fileName)
    
    def on_click_browse_testDataset(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', ".\\" , '*.csv') #QtCore.QDir.rootPath()
        #Assign path to Test Dataset textbox
        self.textbox_testDataset.setText(fileName)

    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
        
    def on_click_compute(self):
        #Get value from textbox3
        textboxValue_numData = self.textbox_numData.text()
        textboxValue_cluster = self.textbox_cluster.text()
        textboxValue_trainDataset = self.textbox_trainDataset.text()
        textboxValue_testDataset = self.textbox_testDataset.text()
        
        #Get value from dropdownlist Norm
        ddlValue_Norm = self.ddl.currentText()
        
        getCluster(self, textboxValue_cluster, textboxValue_trainDataset, textboxValue_testDataset, ddlValue_Norm, textboxValue_numData)
        #self.textbox_cluster.setText("")

def getCluster(self, textboxValue_cluster, textboxValue_trainDataset, textboxValue_testDataset, ddlValue_Norm, textboxValue_numData):
    
    #Start timer (Train)
    train_start = time.time()
    try:
        k = int(textboxValue_cluster)
        maxDataSize = int(textboxValue_numData)
    except:
        messageBox("Error!", "Please enter only integer for cluster and number of data!", 1)
        sys.exit()
    
    #Validation check K more than 10
    if k > 10:
        k = 10
        messageBox("Warning!", "Only value between 1-10 is allowed for Cluster(K). Cluster(K) has been set to 10!", 1)
    if k < 0:
        k = 1
        messageBox("Warning!", "Only value between 1-10 is allowed for Cluster(K). Cluster(K) has been set to 1!", 1)
    if k >= 4 and k <= 10:
        messageBox("Warning!", "K with 4 or more might take some time for centroids to initialize!", 1)
        
    
    #Validation check maxDataSize more than 100    
    if maxDataSize > 100:
        maxDataSize = 100
        messageBox("Warning!", "Maximum number of data is 100. Number of data has been set to 100!", 1)
    
    x_canvasSize, y_canvasSize = 200, 200
    #QMessageBox.question(self, 'Message!', "You typed: " + str(k), QMessageBox.Ok, QMessageBox.Ok)
    
    #Read csv file
    data = panda.read_csv(textboxValue_trainDataset, header=None, usecols=[0,1,2,3])
    data_temp = panda.read_csv(textboxValue_trainDataset, header=None)
    w = np.array(data[0:maxDataSize][0]) #arrayWXYZ[0]
    x = np.array(data[0:maxDataSize][1]) #arrayWXYZ[1]
    y = np.array(data[0:maxDataSize][2]) #arrayWXYZ[2] 
    z = np.array(data[0:maxDataSize][3]) #arrayWXYZ[3]
    arrayWXYZ = np.array(data)
    
    #Initialize variables
    centroidsW = []
    centroidsX = []
    centroidsY = []
    centroidsZ = []
    centW_new = []
    centX_new = []
    centY_new = []
    centZ_new = []
    centroids_new = []
    iterations = 0
    centroids_error = 1 #to proceed into loop
    numP = 0
    reinitializeCentroids = 1
    
    print("Initializing centroids...Please wait...")
    
    while(centroids_error != 0):  
        while(reinitializeCentroids > 0):
        
            #conditions to enter this "if" block
            # - when first entering loop
            # - Initialize centroids if haven't
            # - Re-intialize centroids if centroids contain Nan or 0 (ie. if initialization fails)
            if reinitializeCentroids > 0:
                reinitializeCentroids = 0
                centroidsW = []
                centroidsX = []
                centroidsY = []
                centroidsZ = []
                centroids_new = []
                iterations = 0
                cWXYZ, centroidsW, centroidsX, centroidsY, centroidsZ = initializeCentroids(centroidsW, centroidsX, centroidsY, centroidsZ, w, x, y, z, k)
            
        pointToCluster = []
        averageW = np.zeros(k)
        averageX = np.zeros(k)
        averageY = np.zeros(k)
        averageZ = np.zeros(k)
        numPoints = np.zeros(k)

        for i in range(len(w)): #Accumulate averages of coordinates and number of total points
            if ddlValue_Norm == "L1 Norm":
                dist = L1_norm(arrayWXYZ[i], cWXYZ, 1) # computes euclidean distance (L1 norm) between each data point(w,x,y,z) and k numbers of initialized centroids. Returns an array of k numbers of euclidean distance eg. [5.59642743 5.97662112 3.4525353]
            elif ddlValue_Norm == "L2 Norm":
                dist = L2_norm(arrayWXYZ[i], cWXYZ, 1) # computes euclidean distance (L2 norm) between each data point(w,x,y,z) and k numbers of initialized centroids. Returns an array of k numbers of euclidean distance eg. [5.59642743 5.97662112 3.4525353]

                
            pointToCluster.append((np.argmin(dist), arrayWXYZ[i])) # gets index of computed euclidean distance with minimum/shortest value (eg. among [5.59642743 5.97662112 3.4525353]) and associated coordinates of w,x,y,z and append to array
                
            #print("--pointToCluster-- ")
            #for p in range(len(pointToCluster)):
            #    print("pointToCluster [{0}]: {1}".format(p, pointToCluster[p]))
            averageW[np.argmin(dist)] += arrayWXYZ[i][0]
            averageX[np.argmin(dist)] += arrayWXYZ[i][1]
            averageY[np.argmin(dist)] += arrayWXYZ[i][2]
            averageZ[np.argmin(dist)] += arrayWXYZ[i][3]
            numPoints[np.argmin(dist)] += int(1)
        
        #Validation check - Checks if centroids initialization is poor and needs reinitialization
        numPts_Threshold = math.floor((100/k)/2) # parameter to ensure the minimum # of points per cluster
        numPoints = numPoints.astype(int)
        for y in range(k):
            if numPoints[y] <= numPts_Threshold or np.sum(cWXYZ[y]) == 0:
            #if np.sum(cWXYZ[y]) == 0 or isNan_cWXYZ:
                reinitializeCentroids += 1

        pointToCluster = np.array(pointToCluster)
        
        print("-"*20 + " TRAIN " + "-"*20)
        print("ITERATION {0}".format(iterations+1))
        print("averageW: {0}".format(averageW))
        print("averageX: {0}".format(averageX))
        print("averageY: {0}".format(averageY))
        print("averageZ: {0}".format(averageZ))
        print("numPoints: {0}".format(numPoints))
        print("")
        
        numP = numPoints
        centW_new, centX_new, centY_new, centZ_new = updateCentroids(averageW, averageX, averageY, averageZ, numPoints)
        centroids_new = np.array(list(zip(centW_new, centX_new, centY_new, centZ_new)))
        
        print("previous Centroids: ")
        print(cWXYZ)
        print("Updated Centroids: ")
        print(centroids_new)
        print("-"*50)
        
        #Validation check - Centroids error
        if ddlValue_Norm == "L1 Norm":
            centroids_error = L1_norm(cWXYZ, centroids_new, None) #To determine if centroids needs updating otherwise stop iteration
        elif ddlValue_Norm == "L2 Norm":
            centroids_error = L2_norm(cWXYZ, centroids_new, None) #To determine if centroids needs updating otherwise stop iteration
        print("Centroids error: ")
        print(centroids_error)
        cWXYZ = centroids_new
        iterations += 1
        print()
    
    #Sort points according to their clusters
    ptsClusters = []
    pts = 0
    for m in range(k):
        temp_pts = []
        for y in range(len(pointToCluster)):
            if pointToCluster[y][0] == m:
                temp_pts.append(pointToCluster[y][1])
        ptsClusters.append(temp_pts)
    
    #Plot data points and clusters
    fig = plt.figure(figsize=(x_canvasSize,y_canvasSize))
    ax = fig.add_subplot(111, projection='3d')    
    cmap_array = ['Blues_r', 'Greens_r', 'Reds_r', 'Purples_r', 'Greys_r', 'pink_r', 'Oranges_r', 'jet_r', 'copper_r', 'plasma_R'] # initialize cmap for plotting
    markers = ["P", "v" , "," , "o" , "^" , "<", ">", ".", "1", "p"] # initialize markers for plotting
    
    #Slice points in specific coordinates (4 columns)
    for r in range(k):
        print("CLUSTER {0}: {1}".format(r+1, ptsClusters[r])) #len(pointToCluster[:,1][0]
        print("Total # of points in cluster {0}: {1}".format(r+1, len(ptsClusters[r])))
        w_sliced = []
        x_sliced = []
        y_sliced = []
        z_sliced = []
        for g in range(len(ptsClusters[r])):
            w_sliced.append(ptsClusters[r][g][0])
            x_sliced.append(ptsClusters[r][g][1])
            y_sliced.append(ptsClusters[r][g][2])
            z_sliced.append(ptsClusters[r][g][3])
        
        #plot data points
        img = ax.scatter(w_sliced, x_sliced, y_sliced, c=z_sliced, cmap=cmap_array[r], s=30, marker=markers[r])
        
        #plot k clusters (clusters are ones with bigger sized point)
        ax.scatter(centroids_new[r][0], centroids_new[r][1], centroids_new[r][2], c=centroids_new[r][3], cmap=cmap_array[r], marker=markers[r], s=200)
        ax.set_title("TRAIN (" + ddlValue_Norm + ")", fontsize=40)
        colourBar = fig.colorbar(img)
        colourBar.set_label('Cluster ' + str(r+1) + "(TRAIN)")
        print("Centroid: ")
        print(centroids_new[r][0], centroids_new[r][1], centroids_new[r][2], centroids_new[r][3])
        print("")

    #fig.savefig("kmeans.png")
    plt.show()
    
    #Output csv file with appended column (classification) - Train
    train_fileName = updateCSV(textboxValue_trainDataset, pointToCluster[:,0], data_temp[0:maxDataSize])
    
    #Stop timer (Train)
    train_end = time.time()
    
    
    #Start timer (Test)
    test_start = time.time()
    
    #run on Test dataset
    test_centroids_error, test_centroids_coordinate, test_pointToCluster, test_fileName, centroids_error_overall = predict(textboxValue_testDataset, centroids_new, ddlValue_Norm, k, x_canvasSize, y_canvasSize, cmap_array, markers)
    
    #Stop timer (Test)
    test_end = time.time()
    
    #Write summary to text file and terminal
    writeSummary(train_start, train_end, iterations, centroids_new, pointToCluster, numP, ddlValue_Norm, test_centroids_error, test_centroids_coordinate, test_pointToCluster, train_fileName, test_fileName, test_start, test_end, centroids_error_overall, textboxValue_testDataset)
    
    


def predict(textboxValue_testDataset, centroids_new, ddlValue_Norm, k, x_canvasSize, y_canvasSize, cmap_array, markers):
    #Read csv file
    testData = panda.read_csv(textboxValue_testDataset, header=None, usecols=[0,1,2,3])
    testData_temp = panda.read_csv(textboxValue_testDataset, header=None) #temp variable for updating csv
    w = np.array(testData[:][0]) 
    x = np.array(testData[:][1]) 
    y = np.array(testData[:][2]) 
    z = np.array(testData[:][3])
    test_arrayWXYZ = np.array(testData)

    #Initialize variables
    centroidsW = []
    centroidsX = []
    centroidsY = []
    centroidsZ = []
    centW_new = []
    centX_new = []
    centY_new = []
    centZ_new = []
    centroids_error_array = []
    
    test_pointToCluster = []
    averageW = np.zeros(k)
    averageX = np.zeros(k)
    averageY = np.zeros(k)
    averageZ = np.zeros(k)
    numPoints = np.zeros(k)

    for i in range(len(w)): #Accumulate averages of coordinates and number of total points
        if ddlValue_Norm == "L1 Norm":
            dist = L1_norm(test_arrayWXYZ[i], centroids_new, 1) # computes euclidean distance (L1 norm) between each data point(w,x,y,z) and k numbers of initialized centroids. Returns an array of k numbers of euclidean distance eg. [5.59642743 5.97662112 3.4525353]
        elif ddlValue_Norm == "L2 Norm":
            dist = L2_norm(test_arrayWXYZ[i], centroids_new, 1) # computes euclidean distance (L2 norm) between each data point(w,x,y,z) and k numbers of initialized centroids. Returns an array of k numbers of euclidean distance eg. [5.59642743 5.97662112 3.4525353]

        test_pointToCluster.append((np.argmin(dist), test_arrayWXYZ[i])) # gets index of computed euclidean distance with minimum/shortest value (eg. among [5.59642743 5.97662112 3.4525353]) and associated coordinates of w,x,y,z and append to array
                
        #print("--test_pointToCluster-- ")
        #for p in range(len(test_pointToCluster)):
        #    print("test_pointToCluster [{0}]: {1}".format(p, test_pointToCluster[p]))
        averageW[np.argmin(dist)] += test_arrayWXYZ[i][0]
        averageX[np.argmin(dist)] += test_arrayWXYZ[i][1]
        averageY[np.argmin(dist)] += test_arrayWXYZ[i][2]
        averageZ[np.argmin(dist)] += test_arrayWXYZ[i][3]
        numPoints[np.argmin(dist)] += int(1)
    
    test_pointToCluster = np.array(test_pointToCluster)
    
    print("-"*20 + " PREDICT " + "-"*20)
    print("averageW: {0}".format(averageW))
    print("averageX: {0}".format(averageX))
    print("averageY: {0}".format(averageY))
    print("averageZ: {0}".format(averageZ))
    print("numPoints: {0}".format(numPoints))
    print("")
        
    centW_new, centX_new, centY_new, centZ_new = updateCentroids(averageW, averageX, averageY, averageZ, numPoints)
    test_centroids = np.array(list(zip(centW_new, centX_new, centY_new, centZ_new)))
    
    print("Predicted Centroids: ")
    print(centroids_new)
    print("Test Centroids: ")
    print(test_centroids)
    print("-"*50)
    
    for i in range(k):
        #Validation check - Centroids error
        if ddlValue_Norm == "L1 Norm":
            print("centroids_new[{0}]: {1}".format(i+1, centroids_new[i]))
            print("test_centroids[{0}]: {1}".format(i+1, test_centroids[i]))
            centroids_error = L1_norm(centroids_new[i], test_centroids[i], None) #To determine if centroids needs updating otherwise stop iteration
        elif ddlValue_Norm == "L2 Norm":
            print("centroids_new[{0}]: {1}".format(i+1, centroids_new[i]))
            print("test_centroids[{0}]: {1}".format(i+1, test_centroids[i]))
            centroids_error = L2_norm(centroids_new[i], test_centroids[i], None) #To determine if centroids needs updating otherwise stop iteration
        print("Centroids error: ")
        print(centroids_error)
        centroids_error_array.append(centroids_error)
    centroids_error_overall = L2_norm(centroids_new, test_centroids, None) #To determine if centroids needs updating otherwise stop iteration
    
    #Save CSV file
    test_fileName = updateCSV(textboxValue_testDataset, test_pointToCluster[:,0], testData_temp)
    
    #----------------------------------------PLOT FIGURE----------------------------------------
    #Sort points according to their clusters
    test_ptsClusters = []
    pts = 0
    for m in range(k):
        temp_pts = []
        for y in range(len(test_pointToCluster)):
            if test_pointToCluster[y][0] == m:
                temp_pts.append(test_pointToCluster[y][1])
        test_ptsClusters.append(temp_pts)
    
    #Plot data points and clusters
    fig = plt.figure(figsize=(x_canvasSize,y_canvasSize))
    ax = fig.add_subplot(111, projection='3d')    
    cmap_array = cmap_array # initialize cmap for plotting
    markers = markers # initialize markers for plotting
    
    #Slice points in specific coordinates (4 columns)
    print("="*20 + " TEST " + "="*20)
    for r in range(k):
        print("CLUSTER {0}: {1}".format(r+1, test_ptsClusters[r]))
        print("Total # of points in cluster {0}: {1}".format(r+1, len(test_ptsClusters[r])))
        w_sliced = []
        x_sliced = []
        y_sliced = []
        z_sliced = []
        for g in range(len(test_ptsClusters[r])):
            w_sliced.append(test_ptsClusters[r][g][0])
            x_sliced.append(test_ptsClusters[r][g][1])
            y_sliced.append(test_ptsClusters[r][g][2])
            z_sliced.append(test_ptsClusters[r][g][3])
        
        #plot data points
        img = ax.scatter(w_sliced, x_sliced, y_sliced, c=z_sliced, cmap=cmap_array[r], s=30, marker=markers[r])
        
        #plot k clusters (clusters are ones with bigger sized point)
        ax.scatter(centroids_new[r][0], centroids_new[r][1], centroids_new[r][2], c=centroids_new[r][3], cmap=cmap_array[r], marker=markers[r], s=200)
        ax.set_title("TEST (" + ddlValue_Norm + ")", fontsize=40)
        colourBar = fig.colorbar(img)
        colourBar.set_label('Cluster ' + str(r+1) + " (TEST)")
        print("Centroid: ")
        print(centroids_new[r][0], centroids_new[r][1], centroids_new[r][2], centroids_new[r][3])
        print("")
    plt.show()
    
    return np.array(centroids_error_array), test_centroids, test_pointToCluster, test_fileName, centroids_error_overall

def updateCSV(csv, content, csv_pd):
    fileName = os.path.basename(csv)
    savePath = os.path.dirname(csv)
    file, ext = fileName.split(".")
    fileName_new = savePath + "\\output_" + file + "." + ext
    
    csv_pd[len(csv_pd.columns)] = content
    
    csv_pd.to_csv(fileName_new, index = False, header=False)
    
    print("=================SAVED FILE=================")
    print("Filename: {0}".format(fileName_new))
    print("============================================")
    
    return fileName_new

def messageBox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def get_sublists(original_list, number_of_sub_list_wanted):
    sublists = list()
    for sub_list_count in range(number_of_sub_list_wanted): 
        sublists.append(original_list[sub_list_count::number_of_sub_list_wanted])
    return sublists

def initializeCentroids(centroidsW, centroidsX, centroidsY, centroidsZ, w, x, y, z, k):
    centroidsW = np.array(np.random.randint(0, np.amax(w), size=k)) #gets k random numbers(between 0 and max/highest number in coordinate w) to initialize centroids for coordinate w
    centroidsX = np.array(np.random.randint(0, np.amax(x), size=k)) #gets k random numbers(between 0 and max/highest number in coordinate w) to initialize centroids for coordinate x
    centroidsY = np.array(np.random.randint(0, np.amax(y), size=k)) #gets k random numbers(between 0 and max/highest number in coordinate w) to initialize centroids for coordinate y
    centroidsZ = np.array(np.random.randint(0, np.amax(z), size=k)) #gets k random numbers(between 0 and max/highest number in coordinate w) to initialize centroids for coordinate z
    cWXYZ = np.array(list(zip(centroidsW, centroidsX, centroidsY, centroidsZ))) #Merge centroids coordinate w,x,y,z into list
    """
    print("")
    print("Initialize Centroids: ")
    print(cWXYZ)
    """
    
    return cWXYZ, centroidsW, centroidsX, centroidsY, centroidsZ

def writeSummary(start, end, iterations, centroids_new, pointToCluster, numP, ddlValue_Norm, test_centroids_error, test_centroids_coordinate, test_pointToCluster, train_fileName, test_fileName, test_start, test_end, centroids_error_overall, textboxValue_testDataset):
    #Write to kmeans.txt file
    savePath = os.path.dirname(textboxValue_testDataset)
    f = open(savePath + "\\kmeans.txt","w+")
    f.write("=================TRAIN=================\n")
    f.write("Time taken (s): ")
    f.write(str(end - start))
    f.write("\n")
    f.write("Iterations: " + str(iterations))
    f.write("\n")
    f.write("Norm: {0}".format(ddlValue_Norm))
    f.write("\n")
    f.write("Output file: ")
    f.write(train_fileName)
    f.write("\n")
    
    #output summary to command line
    print("=================TRAIN=================")
    print("Time taken (s): {0}".format((end - start)))
    print("Iterations: " + str(iterations))
    print("Norm: {0}".format(ddlValue_Norm))
    print("Output file: {0}".format(train_fileName))
    print("Latest Centroids: ")
    
    #Centroids Predicted
    f.write("Centroids (Predicted): \n")
    for i in range(len(centroids_new)):
        print("Centroid (Predicted) " + str(i+1) + " = " + str(centroids_new[i][0]) + ", " + str(centroids_new[i][1]) + ", " + str(centroids_new[i][2]) + ", " + str(centroids_new[i][3]))
        
        f.write("Centroid " + str(i+1) + " = ")
        f.write(str(centroids_new[i][0]))
        f.write(", ")
        f.write(str(centroids_new[i][1]))
        f.write(", ")
        f.write(str(centroids_new[i][2]))
        f.write(", ")
        f.write(str(centroids_new[i][3]))
        f.write("\n")
    
    #Number of points    
    print("Number of points: ")
    print(str(numP) + " #" + str(np.sum(numP)))
    
    f.write("Number of points: ")
    f.write(str(numP) + " #" + str(np.sum(numP)))
    f.write("\n")
    
    
    print("Centroids (Cluster - Points): --Please refer to the save file named kmeans.txt--")
    print("=====================================")

    f.write("Centroid (Cluster - Points): \n")
    for i in range(len(pointToCluster)):
        f.write(str(pointToCluster[i][0]))
        f.write(" - ")
        f.write(str(pointToCluster[i][1][0]))
        f.write(", ")
        f.write(str(pointToCluster[i][1][1]))
        f.write(", ")
        f.write(str(pointToCluster[i][1][2]))
        f.write(", ")
        f.write(str(pointToCluster[i][1][3]))
        f.write("\n")
    f.write("======================================")
    
    
    
    f.write("\n=================TEST=================\n")
    f.write("Time taken (s): ")
    f.write(str(test_end - test_start))
    f.write("\n")
    f.write("Norm: {0}".format(ddlValue_Norm))
    f.write("\n")
    f.write("Number of points: ")
    f.write(str(len(test_pointToCluster)))
    f.write("\n")
    f.write("Output file: ")
    f.write(test_fileName)
    f.write("\n")
    
    #output summary to command line
    print("=================TEST=================")
    print("Time taken (s): {0}".format((test_end - test_start)))
    print("Norm: {0}".format(ddlValue_Norm))   
    print("Number of points: {0}".format(len(test_pointToCluster)))
    print("Output file: {0}".format(test_fileName))
    
    #Centroids Predicted
    f.write("Centroids (Predicted): \n")
    for i in range(len(centroids_new)):
        print("Centroid (Predicted)" + str(i+1) + " = " + str(centroids_new[i][0]) + ", " + str(centroids_new[i][1]) + ", " + str(centroids_new[i][2]) + ", " + str(centroids_new[i][3]))
        
        f.write("Centroid " + str(i+1) + " = ")
        f.write(str(centroids_new[i][0]))
        f.write(", ")
        f.write(str(centroids_new[i][1]))
        f.write(", ")
        f.write(str(centroids_new[i][2]))
        f.write(", ")
        f.write(str(centroids_new[i][3]))
        f.write("\n")
    
    #Centroids Test
    f.write("Centroids (Test): \n")
    for j in range(len(centroids_new)):    
        print("Centroid (Test)" + str(j+1) + " = " + str(test_centroids_coordinate[j][0]) + ", " + str(test_centroids_coordinate[j][1]) + ", " + str(test_centroids_coordinate[j][2]) + ", " + str(test_centroids_coordinate[j][3]))
        
        f.write("Centroid " + str(j+1) + " = ")
        f.write(str(test_centroids_coordinate[j][0]))
        f.write(", ")
        f.write(str(test_centroids_coordinate[j][1]))
        f.write(", ")
        f.write(str(test_centroids_coordinate[j][2]))
        f.write(", ")
        f.write(str(test_centroids_coordinate[j][3]))
        f.write("\n")
    
    #Centroids Error    
    f.write("Centroid Errors: \n")
    for j in range(len(test_centroids_error)):    
        print("Centroid (Error)" + str(j+1) + " = " + str(test_centroids_error[j]) + ", " + str(test_centroids_error[j]) + ", " + str(test_centroids_error[j]))
        
        f.write("Centroid " + str(j+1) + " = ")
        f.write(str(test_centroids_error[j]))
        f.write("\n")
    
    f.write("Accuracy: {0}%".format(round((1-(np.sum(test_centroids_error)/len(test_centroids_error)))*100, 2))) #centroids_error_overall
    f.write("\n")
    print("Accuracy: {0}%".format((round((1-(np.sum(test_centroids_error)/len(test_centroids_error)))*100, 2)))) #centroids_error_overall
    
    print("Centroids (Cluster - Points): --Please refer to the save file named kmeans.txt--")
    print("==================================")

    f.write("Centroid (Cluster - Points): \n")
    for i in range(len(test_pointToCluster)):
        f.write(str(test_pointToCluster[i][0]))
        f.write(" - ")
        f.write(str(test_pointToCluster[i][1][0]))
        f.write(", ")
        f.write(str(test_pointToCluster[i][1][1]))
        f.write(", ")
        f.write(str(test_pointToCluster[i][1][2]))
        f.write(", ")
        f.write(str(test_pointToCluster[i][1][3]))
        f.write("\n")
    
    f.write("=====================================")
    
    f.close()
        
def updateCentroids(cw, cx, cy, cz, p):    
    for a in range(len(cw)):
        cw[a] = cw[a]/p[a]
        cx[a] = cx[a]/p[a]
        cy[a] = cy[a]/p[a]
        cz[a] = cz[a]/p[a]
        
    return cw, cx, cy, cz

def L1_norm(p1, p2, axisParam):
    L1 = np.sum(abs(p1-p2), axis=axisParam)
    return L1

def L2_norm(p1, p2, axisParam):
    l2 = np.sqrt(np.sum(np.power((p1 - p2),2), axis=axisParam))
    return l2
    
def eudistance(p1,p2, axisParam):
    return np.linalg.norm((p1-p2), axis=axisParam)

def main():

    app = QApplication(sys.argv)
    
    mainWindow = MainWindow()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()