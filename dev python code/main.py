#!/usr/bin/python3
import random as rd
import matplotlib
matplotlib.use("Qt4Agg", force = True)
from matplotlib.pyplot import *
from math import *
import numpy as np
import skimage as ski
from skimage.exposure import adjust_gamma
from skimage.color import rgb2gray
from scipy import misc
import PIL.ImageOps
style.use('ggplot')
import sys
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
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage import io
from math import sqrt
from skimage.color import rgb2gray
from scipy import misc
import pandas as pd
from collections import Counter
from skimage import exposure
from skimage import transform
from sklearn.decomposition import PCA
from numpy import fft
from texture import *
from sklearn.ensemble import RandomForestClassifier
from vectorize import *
from dispersionratio import *



form_class = uic.loadUiType("bycells.ui")[0]
form_class2 = uic.loadUiType("bycells_classwindow.ui")[0]

class MainWindow(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.cellpoints = np.array([])
        self.FindCells.clicked.connect(self.Id_cells)
        self.Classify.clicked.connect(self.start_clicked)
        self.AddClassified.clicked.connect(self.create_csv)
        self.validatebutton.clicked.connect(self.validateAutoClass)
        self.imageviewbutton.clicked.connect(self.openMainFig)
        self.autoClassButton.clicked.connect(self.AutoClassification)
        self.Boxsize.setText("101")
        self.fig = Figure()
        self.THEimage = np.array([])
        self.BLUEimage = 0
        self.THEblobs = np.array([])
        self.DatabaseSize.setText(str( len(glob.glob('singleCells/*.png') ) ) )
        self.table.setColumnCount(3)
        self.layout.addWidget(self.table, 1, 0)
        self.table.setHorizontalHeaderLabels(['index', 'auto class', 'gold class'])
        self.dirButton.clicked.connect(self.chooseDirecoty)
        self.directory = 'singleCells/'
        self.saveDir.setText('singleCells/')

    def removeCell(self):
        cellnumber = int(self.rmvCellN.text())
        self.THEblobs[cellnumber:-1] = self.THEblobs[cellnumber+1:]
        self.THEblobs = self.THEblobs[:-1]
        self.nMarkedCells.setText(str(int(self.nMarkedCells.text() )-1 ) )
        self.table.removeRow(cellnumber)
        for i in range(len(self.THEblobs)):
            self.table.setItem(i, 0, QtGui.QTableWidgetItem(str(i)))
        self.ImgAddPatches()
        self.rmvCellN.setText('')

    def chooseDirecoty(self):
        directory = QtGui.QFileDialog.getExistingDirectory(self)
        self.saveDir.setText(str(directory) + '/')
        self.DatabaseSize.setText(str( len(glob.glob(str(self.saveDir.text())+ '*.png') ) ) )

    def create_csv(self):
        savename = str(self.saveNames.text())
        filenames  = np.array([savename+str(self.table.item(i,0).text())+'.png' for i in range(int(self.nMarkedCells.text() ) ) ])
        #filenamesList = filenames.tolist()
        classnames = np.array([         str(self.table.item(i,2).text())        for i in range(int(self.nMarkedCells.text() ) ) ])
        classtable = pd.DataFrame( np.transpose(np.vstack((filenames,classnames))))#, index=dates, columns=[nome , classe])
        print(classtable)
        saveclassification = classtable.to_csv(str(self.saveDir.text())+ savename +'class.csv',index=False,header=['file','class'])
        self.DatabaseSize.setText(str( len(glob.glob(str(self.saveDir.text())+'*.png') ) ) )

    def save_crops(self):
        squaresize = int(str(self.Boxsize.text()))
        savename = str(self.saveNames.text())
        blobs = self.THEblobs
        for number, blob in enumerate(blobs):
            y, x, r = blob
            y = y +squaresize #adjusting centers
            x = x +squaresize #DONT ASK ME WHY

            crop = self.THEimage[int(y)-int(squaresize/2) : int(y)+int(squaresize/2), int(x)-int(squaresize/2) : int(x)+int(squaresize/2)]
            io.imsave(str(self.saveDir.text())+ savename +str(number)+'.png', crop)


    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
        if event.button == 3:
            squaresize = int(str(self.Boxsize.text()))
            self.THEblobs =np.array(self.THEblobs.tolist() + [[int(event.ydata - squaresize), int(event.xdata - squaresize), int(str(self.Boxsize.text()))]])
            print(self.THEblobs)
            self.table.setHorizontalHeaderLabels(['index', 'auto class', 'gold class'])
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)
            self.table.setItem(rowPosition , 0, QtGui.QTableWidgetItem(str(rowPosition)))
            self.table.setItem(rowPosition , 1, QtGui.QTableWidgetItem("-"))
            self.table.setItem(rowPosition , 2, QtGui.QTableWidgetItem("-"))
            self.nMarkedCells.setText(str(int(self.nMarkedCells.text()) + 1))
            self.ImgAddPatches()
        elif event.button == 2:
            print(self.THEblobs[:,0:2])
            dist = np.sum((self.THEblobs[:,0:2]+101-[event.ydata,event.xdata])**2,1)
            if min(dist) < 800:
                line = dist.tolist().index(min(dist))
                print(line)
                self.rmvCellN.setText(str(line))
                self.removeCell()



    def Id_cells(self):
        squaresize = int(str(self.Boxsize.text()))
        image_gray = self.BLUEimage
        blobs = blob_dog(image_gray[squaresize:-squaresize,squaresize:-squaresize], min_sigma = 10, max_sigma = 30, threshold=.8)
        self.THEblobs = blobs
        self.nMarkedCells.setText(str(len(blobs)))
        self.table.setRowCount(len(blobs))
        self.table.setColumnCount(3)
        self.layout.addWidget(self.table, 1, 0)
        self.table.setHorizontalHeaderLabels(['index', 'auto class', 'gold class'])
        self.ImgAddPatches()

    def ImgAddPatches(self):
        squaresize = int(str(self.Boxsize.text()))
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


    def start_clicked(self):
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

    def coloring_types(self, cellClassList):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
        color_dict = dict(zip([x[0] for x in Counter(cellClassList).most_common()],colors))
        self.fig, ax = subplots(1, 1)
        squaresize = int(str(self.Boxsize.text()))
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

    def AutoClassification(self):
        self.save_crops()
        #open images
        training = pd.DataFrame()
        testImg = pd.DataFrame()
        list_ = []
        allFiles = glob.glob(str(self.saveDir.text())+"*.csv")
        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=None, header=0)
            list_.append(df)
        training = pd.concat(list_)
        training["photo"] = training.file.apply(lambda x: misc.imread(str(self.saveDir.text()) + x))
        training["photo"] = training.photo.apply(rgb2gray)
        training["photo"] = training.photo.apply(exposure.equalize_adapthist)
        testImg["files"] = glob.glob(str(self.saveDir.text())+ str(self.saveNames.text()) +"*.png")
        testImg["photo"] = [misc.imread(x) for x in glob.glob(str(self.saveDir.text())+ str(self.saveNames.text()) +"*.png")]
        testImg["photo"] = testImg.photo.apply(rgb2gray)
        testImg["photo"] = testImg.photo.apply(exposure.equalize_adapthist)


        # Rotate training images
        def rotate(df, degrees):
            result = df.copy()
            result.photo = result.photo.apply(lambda x: transform.rotate(x, degrees))
            return result

        number_of_rotations = 20
        orig_training = training.copy()
        for i in [(360./number_of_rotations) * (i+1) for i in range(number_of_rotations)]:
            training = pd.concat((training, rotate(orig_training, i)))

        # Initialize features with texture values
        train_feats = np.array([x for x in training.photo.apply(texture).values])
        print(train_feats)
        Y_training = training["class"].values
        testImg_feats = np.array([x for x in testImg.photo.apply(texture).values])

        # Add dispersion ratios to features
        training["dispersion"] = training.photo.apply(dispersionratio)
        train_feats = np.hstack( ( train_feats, np.array([x for x in training["dispersion"].values]).reshape(-1,1) ) )
        testImg["dispersion"] = testImg.photo.apply(dispersionratio)
        testImg_feats  = np.hstack( ( testImg_feats, np.array([x for x in testImg["dispersion"].values]).reshape(-1,1) ) )


        # Apply FFT to photos (does NOT add to features yet)
        training["FFT"] = training.photo.apply(fft.fft2)
        training["FFT"] = training.FFT.apply(abs)
        training["Phase"] = training.FFT.apply(np.angle)
        testImg["FFT"] = testImg.photo.apply(fft.fft2)
        testImg["FFT"] = testImg.FFT.apply(abs)
        testImg["Phase"] = testImg.FFT.apply(np.angle)

        # Dimensionality reduction on the FFTs
        pca = PCA(n_components = 15)
        pcb = PCA(n_components = 15)
        fabsPCA = pca.fit(vectorize(training["FFT"]))
        fphiPCA = pcb.fit(vectorize(training["Phase"]))

        #Adding ffts to feature set
        train_feats_final = np.hstack( ( train_feats, fphiPCA.transform(vectorize(training["Phase"])), fabsPCA.transform(vectorize(training["FFT"])) ) )
        testImg_feats_final = np.hstack( ( testImg_feats, fphiPCA.transform(vectorize(testImg["Phase"])), fabsPCA.transform(vectorize(testImg["FFT"])) ) )

        train_feats_final = normalize_columns(train_feats_final)
        testImg_feats_final = normalize_columns(testImg_feats_final)
        clf = RandomForestClassifier(n_estimators = 9)
        print(len(train_feats_final), len(Y_training))
        clf.fit(np.nan_to_num(train_feats_final), np.nan_to_num(Y_training) )
        Y_predict = clf.predict(np.nan_to_num(testImg_feats_final))
        for number, cellclass in enumerate(Y_predict):
            self.table.setItem(number , 1, QtGui.QTableWidgetItem(str(cellclass)))

        self.coloring_types(Y_predict)





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
