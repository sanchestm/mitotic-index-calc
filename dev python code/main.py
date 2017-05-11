#!/usr/bin/python2
import matplotlib
matplotlib.use("Qt4Agg", force = True)
from matplotlib.pyplot import *
from skimage.exposure import adjust_gamma
from skimage.color import rgb2gray
from scipy import misc
import PIL.ImageOps
style.use('ggplot')
from sys import argv
import pickle
import glob
from PyQt4 import QtCore, QtGui, uic
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from ast import literal_eval as make_tuple
import re
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from skimage.feature import blob_dog
from skimage import io
from math import sqrt
from scipy import misc
from collections import Counter
from skimage import exposure
from skimage import transform
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from numpy import array, transpose, vstack, hstack
from numpy import sum, angle, nan_to_num
from pandas import DataFrame, read_csv, concat
from subprocess import check_output
from time import time
from HOGnLBPfeatures import getLBP, getHoG, abs_pha_fft, Get_all_features
from metriclearning import suvrel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from Suvrellib import Suvrel
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from sklearn.feature_extraction.image import img_to_graph
from ImageManagement import get_all_image_names, CreateIndividualCellPatches, checksizephoto



form_class = uic.loadUiType("bycells.ui")[0]
form_class2 = uic.loadUiType("bycells_classwindow.ui")[0]
form_class3 = uic.loadUiType("bycells_directories.ui")[0]

class MainWindow(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.cellpoints = np.array([])
        self.FindCells.clicked.connect(self.Id_cells)
        self.Classify.clicked.connect(self.Classifiers)
        self.AddClassified.clicked.connect(self.create_csv)
        self.validatebutton.clicked.connect(self.validateAutoClass)
        self.imageviewbutton.clicked.connect(self.openMainFig)
        self.cropsize = 100
        self.fig = Figure()
        self.THEimage = np.array([])
        self.BLUEimage = 0
        self.THEblobs = np.array([])
        self.DatabaseSize.setText(str( len(glob.glob('singleCells/*.png') ) ) )
        self.table.setColumnCount(3)
        self.layout.addWidget(self.table, 1, 0)
        self.table.setHorizontalHeaderLabels(['index', 'auto class', 'manual class'])
        self.dirButton.clicked.connect(self.chooseDirectory)
        self.directory = 'singleCells/'
        self.saveDir.setText('singleCells/')
        self.dirWindow.clicked.connect(self.openDIRwindow)

    def openDIRwindow(self):
        dirwindow = allDirectoriesWindow(self)
        dirwindow.exec_()

    def removeCell(self, cellnumber):
        self.THEblobs[cellnumber:-1] = self.THEblobs[cellnumber+1:]
        self.THEblobs = self.THEblobs[:-1]
        self.nMarkedCells.setText(str(int(self.nMarkedCells.text() )-1 ) )
        self.table.removeRow(cellnumber)
        for i in range(len(self.THEblobs)):
            self.table.setItem(i, 0, QtGui.QTableWidgetItem(str(i)))
        self.ImgAddPatches()


    def chooseDirectory(self):
        directory = QtGui.QFileDialog.getExistingDirectory(self)
        self.saveDir.setText(str(directory) + '/')
        self.DatabaseSize.setText(str( len(glob.glob(str(self.saveDir.text())+ '*.png') ) ) )


    def openMainFig(self):
        if self.THEimage.any() == True:
            self.rmmpl()
            self.THEimage = np.array([])
            self.BLUEimage = 0
            for i in range(len(self.THEblobs)):
                self.table.removeRow(0)
            self.nMarkedCells.setText(str(0))
            self.THEblobs = np.array([])

        name = QtGui.QFileDialog.getOpenFileName(self, 'Single File', '~/Desktop/', "Image files (*.jpg *.png *.tif)")
        image = misc.imread(str(name))
        self.saveNames.setText(str(name).split("/")[-1][:-4] + 'i')
        self.THEimage = image
        self.BLUEimage = image[:,:,2]
        baseimage = self.fig.add_subplot(111)
        baseimage.axis('off')
        baseimage.grid(False)
        baseimage.imshow(image)
        self.canvas = FigureCanvas(self.fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.widget, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
        if event.button == 3:
            squaresize = self.cropsize
            self.THEblobs =np.array(self.THEblobs.tolist() + [[int(event.ydata - squaresize), int(event.xdata - squaresize), self.cropsize]])
            #print(self.THEblobs)
            self.table.setHorizontalHeaderLabels(['index', 'auto class', 'manual class'])
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)
            self.table.setItem(rowPosition , 0, QtGui.QTableWidgetItem(str(rowPosition)))
            self.table.setItem(rowPosition , 1, QtGui.QTableWidgetItem("-"))
            self.table.setItem(rowPosition , 2, QtGui.QTableWidgetItem("-"))
            self.nMarkedCells.setText(str(int(self.nMarkedCells.text()) + 1))
            self.ImgAddPatches()
        elif event.button == 2:
            #print(self.THEblobs[:,0:2])
            dist = np.sum((self.THEblobs[:,0:2]+101-[event.ydata,event.xdata])**2,1)
            if min(dist) < 800:
                line = dist.tolist().index(min(dist))
                #print(line)
                self.removeCell(line)

    def changeFIGURE(self, newFIG):
        self.rmmpl()
        self.canvas = FigureCanvas(newFIG)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.widget, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)


    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def Id_cells(self):
        squaresize = self.cropsize
        image_gray = self.BLUEimage
        blobs = blob_dog(image_gray[squaresize:-squaresize,squaresize:-squaresize], min_sigma = 10, max_sigma = 30, threshold=.8)
        self.THEblobs = blobs
        self.nMarkedCells.setText(str(len(blobs)))
        self.table.setRowCount(len(blobs))
        self.table.setColumnCount(3)
        self.layout.addWidget(self.table, 1, 0)
        self.table.setHorizontalHeaderLabels(['index', 'auto class', 'manual class'])
        self.ImgAddPatches()

    def ImgAddPatches(self):
        squaresize = self.cropsize
        self.fig, ax = subplots(1, 1)
        ax.imshow(self.THEimage)
        ax.grid(False)
        ax.axis('off')
        for number, blob in enumerate(self.THEblobs):
            y, x, r = blob
            c = Rectangle((x + int(squaresize/2), y + int(squaresize/2)),squaresize,squaresize, color='r', linewidth=2, alpha = 0.3)
            ax.add_patch(c)
            ax.text(x+squaresize-25,y+ squaresize+25, str(number), color = 'white')
            self.table.setItem(number, 0, QtGui.QTableWidgetItem(str(number)))
            self.table.setItem(number, 1, QtGui.QTableWidgetItem('-'))
            self.table.setItem(number, 2, QtGui.QTableWidgetItem('-'))
        self.changeFIGURE(self.fig)


    def coloring_types(self, cellClassList):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
        color_dict = dict(zip([x[0] for x in Counter(cellClassList).most_common()],colors))
        self.fig, ax = subplots(1, 1)
        squaresize = self.cropsize
        ax.imshow(self.THEimage)
        ax.grid(False)
        ax.axis('off')
        self.layout.addWidget(self.table, 1, 0)
        for number, blob in enumerate(self.THEblobs):
            y, x, r = blob
            c = Rectangle((x + int(squaresize/2), y + int(squaresize/2)),squaresize,squaresize, color=color_dict[cellClassList[number]], linewidth=2, alpha = 0.3)
            ax.add_patch(c)
            ax.text(x+squaresize-25,y+ squaresize+25, str(number), color = 'white')
        self.changeFIGURE(self.fig)

    def Classifiers(self):
        if self.radioButton.isChecked() == True: self.AutoClassification()
        else:                               self.ManualClassification()

    def ManualClassification(self):
        self.save_crops()
        with open('bytemp','w') as f:
            f.write(str(self.saveNames.text())+','+ str(self.nMarkedCells.text())  + ',' + str(self.saveDir.text()))

        classwindow = ManualClassifyWindow(self)
        classwindow.exec_()
        with open('bytemp','r') as f:
            classifiedCells = f.readline().split(',')[:-1]
            self.MitoticIndex.setText( str(classifiedCells.count('mitose')) + '/' + str( classifiedCells.count('mitose') + classifiedCells.count('interfase') ) )
            for i, clasf in enumerate(classifiedCells):
                self.table.setItem(i, 2, QtGui.QTableWidgetItem(clasf) )
        self.coloring_types(classifiedCells)

    def AutoClassification(self):
        ##predicting
        self.save_crops()
        Files2classify = glob.glob(str(self.saveDir.text())+ str(self.saveNames.text())+"*.png")
        images2classifyDF = DataFrame( Files2classify, columns = ["Filename"])
        images2classifyDF["photo"] = images2classifyDF.Filename.apply(lambda x: rgb2gray(misc.imread(x)))
        images2classifyDF["features"] = images2classifyDF.photo.apply(lambda x : np.hstack([abs_pha_fft(x),getHoG(x), getLBP(x)]))
        CLF = pickle.load(open("picklePy3Clf", "rb"))
        images2classifyDF["predict"] =  images2classifyDF.features.apply(lambda x: CLF.predict(x.reshape(1,-1)))

        ##adjusting table
        predicted_list  = np.array(images2classifyDF["predict"].values)
        predicted_list = [i[0] for i in predicted_list]
        print(predicted_list)
        for i, value in enumerate(predicted_list):
            if value == 1:self.table.setItem(i, 1, QtGui.QTableWidgetItem("mitose"))
            else:self.table.setItem(i, 1, QtGui.QTableWidgetItem("interfase"))
        self.MitoticIndex.setText(str(sum(predicted_list)) + '/'+ str(len(predicted_list)+1))

        ##coloring
        self.coloring_types(predicted_list)

    def validateAutoClass(self):
        conta = 0
        if str(self.table.item(int(int(self.nMarkedCells.text())/2 ),1).text()) != '-':
            for i in range(int(self.nMarkedCells.text() ) ):
                if str(self.table.item(i,1).text()) == str(self.table.item(i,2).text()): conta +=1
            print(conta/int(self.nMarkedCells.text() ))
            self.SucessRate.setText(str(   int(float(conta)/int(self.nMarkedCells.text())*100) )+ '%')

        if str(self.table.item(int(int(self.nMarkedCells.text())/2),2).text()) == '-':
            mitose = 0
            interfase = 0
            for i in range(int(self.nMarkedCells.text() ) ):
                self.table.setItem(i, 2, QtGui.QTableWidgetItem(str(self.table.item(i,1).text())))
                if str(self.table.item(i,1).text()) == 'mitose': mitose += 1
                elif str(self.table.item(i,1).text()) == 'interfase': interfase += 1
            self.MitoticIndex.setText(str(mitose) + '/'+ str(mitose + interfase))
            self.SucessRate.setText("100%")

    def create_csv(self):
        savename = str(self.saveNames.text())
        filenames  = np.array([savename+str(self.table.item(i,0).text())+'.png' for i in range(int(self.nMarkedCells.text() ) ) ])
        classnames = np.array([         str(self.table.item(i,2).text())        for i in range(int(self.nMarkedCells.text() ) ) ])
        classtable = DataFrame( np.transpose(np.vstack((filenames,classnames))))#, index=dates, columns=[nome , classe])
        print(classtable)
        saveclassification = classtable.to_csv(str(self.saveDir.text())+ savename +'class.csv',index=False,header=['file','class'])
        self.DatabaseSize.setText(str( len(glob.glob(str(self.saveDir.text())+'*.png') ) ) )

    def save_crops(self):
        squaresize = self.cropsize
        savename = str(self.saveNames.text())
        blobs = self.THEblobs
        for number, blob in enumerate(blobs):
            y, x, r = blob
            y = y +squaresize #adjusting centers
            x = x +squaresize #DONT ASK ME WHY

            crop = self.THEimage[int(y)-int(squaresize/2) : int(y)+int(squaresize/2), int(x)-int(squaresize/2) : int(x)+int(squaresize/2)]
            io.imsave(str(self.saveDir.text())+ savename +str(number)+'.png', crop)



class allDirectoriesWindow(QtGui.QDialog, form_class3):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.addDir.clicked.connect(lambda x: self.addDirectory(DataFrame()) )
        self.AutoClassDirButon.clicked.connect(lambda x: self.DirectoryAutoClassification( int(self.SelectedDirNumber.value())-1))
        self.saveDirs.clicked.connect(self.saveDirectories)
        self.allDirs.setColumnCount(3)
        self.tableLayout.addWidget(self.allDirs)
        self.allDirs.setHorizontalHeaderLabels(['directory', 'Number of imgs', 'MitoticIndex (Mitosis|Interfasis)'])
        if len( glob.glob('directories') )==1:
            directorytable = read_csv('directories')
            for index, row in directorytable.iterrows():
                self.addDirectory(row)

    def addDirectory(self, row):
        if row.empty == True:
            directory = QtGui.QFileDialog.getExistingDirectory(self)
            n_images = "-"
            rowMitoticIndex = "-"
        else:
            directory, n_images, rowMitoticIndex = row["directory"], row["imageNumber"], row["MitoticIndex"]

        self.allDirs.setHorizontalHeaderLabels(['directory', 'Nº of imgs', 'MitoticIndex (Mitosis|Interfasis)'])
        rowPosition = self.allDirs.rowCount()
        self.allDirs.insertRow(rowPosition)
        self.allDirs.setItem(rowPosition , 0, QtGui.QTableWidgetItem(str(directory)) )
        self.allDirs.setItem(rowPosition , 1, QtGui.QTableWidgetItem(str(n_images)) )
        self.allDirs.setItem(rowPosition , 2, QtGui.QTableWidgetItem(str(rowMitoticIndex)) )
        self.allDirs.resizeColumnToContents(0)
        self.N_dir.setText(str(int(self.N_dir.text()) + 1) )

    def saveDirectories(self):
        savename = 'directories'
        dirtable = DataFrame()
        dirtable["directory"] = np.array([ str(self.allDirs.item(i,0).text())  for i in range(int(self.N_dir.text() ) ) if str(self.allDirs.item(i,0).text()) !=''  ])
        dirtable["imageNumber"] = np.array([ str(self.allDirs.item(i,1).text())  for i in range(int(self.N_dir.text() ) ) if str(self.allDirs.item(i,0).text()) !=''  ])
        dirtable["MitoticIndex"] = np.array([ str(self.allDirs.item(i,2).text())  for i in range(int(self.N_dir.text() ) ) if str(self.allDirs.item(i,0).text()) !=''  ])

        dirtable.to_csv(savename,index=False,header=True)
        self.close()

    def DirectoryAutoClassification(self, OpenDirectoryNumber):
        DirectoryPath = str(self.allDirs.item(OpenDirectoryNumber,0).text())
        DirectoryImageNameList = get_all_image_names(DirectoryPath)

        #Obtaining features
        ImgsOpenFileDF = DataFrame(DirectoryImageNameList, columns = ["Filename"])
        ImgsOpenFileDF["photo"] = ImgsOpenFileDF.Filename.apply(lambda x: misc.imread(x))
        ImgsOpenFileDF = ImgsOpenFileDF[ImgsOpenFileDF.photo.apply(checksizephoto)]
        ImgsOpenFileDF["photoblue"] = ImgsOpenFileDF.photo.apply(lambda x : x[:,:,2])
        ImgsOpenFileDF["photo"] = ImgsOpenFileDF.photo.apply(rgb2gray)
        ImgsOpenFileDF["blobs"]= ImgsOpenFileDF.photoblue.apply(lambda x:blob_dog(x[56:-56,56:-56], min_sigma = 10, max_sigma = 30, threshold=.8))
        ImgsOpenFileDF["IndividualCellPatches"] = ImgsOpenFileDF.apply(CreateIndividualCellPatches, axis = 1)
        ImgsOpenFileDF["Features4EachPatch"] =ImgsOpenFileDF.IndividualCellPatches.apply(lambda x : [Get_all_features(patch) for patch in x])

        #classifying
        CLF = pickle.load(open("picklePy3Clf", "rb"))
        def PredictPatch(PatchFeatures):
            return  CLF.predict(PatchFeatures.reshape(1,-1))

        ImgsOpenFileDF["predict"] = ImgsOpenFileDF["Features4EachPatch"].apply(lambda x: array([PredictPatch(patchfeat) for patchfeat in x]).reshape(1,-1))

        #adding mitotic index
        ImgsOpenFileDF["MitoticIndex"] = ImgsOpenFileDF.predict.apply(lambda x: "{0:.2f} ({1}|{2})".format(x.sum()/len(x[0]), x.sum(), len(x[0])- x.sum()))
        ImgsOpenFileDF["Mitosis"] = ImgsOpenFileDF.predict.apply(lambda x : x.sum())
        ImgsOpenFileDF["interfase"] = ImgsOpenFileDF.predict.apply(lambda x:  len(x[0])- x.sum())
        a = ImgsOpenFileDF.Mitosis.values.sum()
        b = ImgsOpenFileDF.interfase.values.sum()
        print("the mitotic index is {0:.2f}, with {1} of {2} cell on mitosis".format(a/(a+b) ,int(a),int(a+b)))


        #updating table
        self.allDirs.setItem(OpenDirectoryNumber, 1, QtGui.QTableWidgetItem(str(ImgsOpenFileDF.shape[0])) )
        self.allDirs.setItem(OpenDirectoryNumber , 2, QtGui.QTableWidgetItem("{0:.2f} ({1}|{2})".format(a/(a+b) ,int(a),int(a+b))) )




class ManualClassifyWindow(QtGui.QDialog, form_class2):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        with open('bytemp','r') as f:
            line = f.readline().split(',')
        self.name=line[0]
        self.ncells=int(line[1])
        self.nameDir = line[2]
        self.index = 0
        self.showCell()
        self.interfasebutton.clicked.connect(self.interfaseClass)
        self.mitosebutton.clicked.connect(self.mitoseClass)
        self.notcell.clicked.connect(self.notcellClass)
        self.unknownButton.clicked.connect(self.unknownClass)
        self.unclassifiableButton.clicked.connect(self.unclassifiableClass)
        self.saveclose.clicked.connect(self.save_n_close)
        self.goldclasses = []

        basemap1 = QtGui.QPixmap('basetransp.png')
        basemap2 = basemap1.scaled(321, 261, QtCore.Qt.KeepAspectRatio)
        self.basecells.setPixmap(basemap2)

    def showCell(self):
        if self.index==self.ncells:
            self.save_n_close()
        else:
            name = self.nameDir+ self.name + str(self.index) +'.png'
            print(name)
            pixmap1 = QtGui.QPixmap(str(name))
            pixmap2 = pixmap1.scaled(451, 261, QtCore.Qt.KeepAspectRatio)
            self.cellimg.setPixmap(pixmap2)
            self.numcell.setText('cell number: '+str(self.index))

    def interfaseClass(self):
        self.goldclasses.append('interfase,')
        self.index+=1
        self.showCell()

    def mitoseClass(self):
        self.goldclasses.append('mitose,')
        self.index+=1
        self.showCell()

    def notcellClass(self):
        self.goldclasses.append('not a cell,')
        self.index+=1
        self.showCell()

    def unknownClass(self):
        self.goldclasses.append('unknown,')
        self.index+=1
        self.showCell()

    def unclassifiableClass(self):
        self.goldclasses.append('unclassifiable,')
        self.index+=1
        self.showCell()

    def save_n_close(self):
        with open('bytemp','w') as f:
            for line in self.goldclasses:
                f.write(line)
        self.close()

app = QtGui.QApplication(sys.argv)
myWindow = MainWindow()
myWindow.show()
app.exec_()
