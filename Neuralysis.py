from PyQt6.QtWidgets import QToolButton, QApplication, QWidget, QMainWindow,QPushButton,QLabel, QToolBar, QStatusBar,QLineEdit, QMessageBox, QCheckBox, QInputDialog, QDialog, QDialogButtonBox, QVBoxLayout, QGridLayout, QComboBox
from PyQt6.QtCore import QSize, Qt,QPoint
from PyQt6.QtGui import QAction, QDragEnterEvent, QIcon, QPixmap, QCursor, QPainter, QColor, QPen, QBrush,QIntValidator, QKeyEvent, QDoubleValidator, QFont

import numpy as np

import torch.nn as nn
import torch.nn.functional as tf
import torchvision.transforms as tt
import torch
import torch.optim as optim

from PIL import Image
import os
import idx2numpy

# Only needed for access to command line arguments
import sys

def orientation(p0,p1,p):
    return (p[1] - p0[1])*(p1[0] - p0[0]) - (p[0] - p0[0])*(p1[1] - p0[1])

##toolbar classes for better organisation

#Joint class
class Joint():
    ##remove need for side
    def __init__(self,parent,pos,side):
        self.parent = parent
        self.pos = pos
        self.side = side
        self.activated = False
        self.controller = self.parent.parent.parent
    
    def draw(self,pos):
        #ellipse drawn from top left corner, not center!!!
        if self.activated == False:
            self.controller.pen.setColor(QColor(38,133,18)) 
            self.controller.brush.setColor(QColor(120,234,96))
        else:
            self.controller.pen.setColor(QColor(102,106,180)) 
            self.controller.brush.setColor(QColor(200,191,231))
        self.controller.painter.setPen(self.controller.pen)
        self.controller.painter.setBrush(self.controller.brush)
        self.controller.painter.drawEllipse(int(pos[0]*np.power(1.2,self.controller.ZOOM) - self.controller.POS[0]) + 120,int(pos[1]*np.power(1.2,self.controller.ZOOM) - self.controller.POS[1]) + 24, int(12*np.power(1.2,self.controller.ZOOM)), int(12*np.power(1.2,self.controller.ZOOM)))

    def isInside(self,point):
            center = np.array([6,6]) + self.pos
            #Only circles for now!
            if (point[0] - center[0])**2 + (point[1] - center[1])**2 <= 36:
                return True
            else:
                return False

#if two points - circle of form (center, radius). If more points - convex polygon 
#position of left and right (input vs output)
class Layer():
    def __init__(self, points, color, left, right, parent,name,params):
        self.points = points
        self.name = name
        self.params = params
        self.mouseOn = False
        self.color = color
        self.parent = parent
        self.controller = self.parent.parent
        #a pair of floats to avoid rounding errors upon zoom. These are global coordinates at zoom level 0
        self.pos = self.controller.POS*np.power(1.2,-self.controller.ZOOM) - self.center() + np.array([480-60,240-12])*np.power(1.2,-self.controller.ZOOM)
        if type(left) != type(None):
            self.left = Joint(self,self.pos + left,"left")
        else:
            self.left = None
        if type(right) != type(None):
            self.right = Joint(self,self.pos + right,"right")
        else:
            self.right = None

    def center(self):
        if len(self.points) == 2:
            return self.points[1]/2
        else:
            return sum(self.points)/len(self.points)

    def updatePos(self,pos):
        delta = pos - self.pos
        self.pos = pos
        if self.left: self.left.pos = self.left.pos + delta
        if self.right: self.right.pos = self.right.pos + delta

    #Absolute position of object relative to origin at zoom level 0! (integer)
    def draw(self,pos):
        delta = pos - self.pos
        #Lines
        self.controller.pen.setColor(QColor(94,94,94)) 
        self.controller.brush.setColor(QColor(94,94,94))
        self.controller.painter.setPen(self.controller.pen)
        self.controller.painter.setBrush(self.controller.brush)
        ##split into two draw - has error
        if self.left:
            rectPos = self.left.pos + np.array([6,4])
        else:
            rectPos = np.array([(self.pos + self.center())[0],self.right.pos[1] + 4])
        self.controller.painter.drawRect(int((rectPos[0] + delta[0])*np.power(1.2,self.controller.ZOOM)-self.controller.POS[0]) + 120,int((rectPos[1] + delta[1])*np.power(1.2,self.controller.ZOOM) -self.controller.POS[1]) + 24, int((self.right.pos[0] - rectPos[0])*np.power(1.2,self.controller.ZOOM)) , int(4*np.power(1.2,self.controller.ZOOM)))
        #Joints
        if self.left: self.left.draw(self.left.pos + delta)
        if self.right: self.right.draw(self.right.pos + delta)
        #Body
        if self.mouseOn == True:
            self.controller.pen.setColor(QColor(0,0,255))
            self.controller.pen.setWidth(1)
            self.controller.painter.setPen(self.controller.pen)
            screenPos = np.power(1.2,self.controller.ZOOM)*pos - self.controller.POS + [120,24] + np.power(1.2,self.controller.ZOOM)*self.center() + [-100, np.power(1.2,self.controller.ZOOM)*self.center()[1] + 12]
            self.controller.painter.drawText(int(screenPos[0]), int(screenPos[1]),200,48, Qt.AlignmentFlag.AlignHCenter, self.name + str(self.params))
            self.controller.pen.setColor(QColor(255,255,255))
            self.controller.pen.setWidth(2)
        else:
            self.controller.pen.setColor(self.color) 
        self.controller.brush.setColor(self.color)
        self.controller.painter.setPen(self.controller.pen)
        self.controller.painter.setBrush(self.controller.brush)
        if len(self.points) > 2:
            self.controller.painter.drawPolygon([QPoint(int((i[0]+ pos[0])*np.power(1.2,self.controller.ZOOM) - self.controller.POS[0]),int((i[1]+ pos[1])*np.power(1.2,self.controller.ZOOM) - self.controller.POS[1])) + QPoint(120,24) for i in self.points])
        elif len(self.points) == 2:
            self.controller.painter.drawEllipse(int((pos[0] + self.points[0][0])*np.power(1.2,self.controller.ZOOM) -self.controller.POS[0]) + 120, int((pos[1] + self.points[0][1])*np.power(1.2,self.controller.ZOOM) -self.controller.POS[1]) + 24, int(self.points[1][0]*np.power(1.2,self.controller.ZOOM)), int(self.points[1][1]*np.power(1.2,self.controller.ZOOM)))
        self.controller.pen.setWidth(1)

    def isInside(self,point):
        if len(self.points) == 2:
            center = self.points[1]/2 + self.pos
            #Only circles for now!
            if (point[0] - center[0])**2 + (point[1] - center[1])**2 <= (self.points[1][0]/2)**2:
                return True
            else:
                return False
        elif len(self.points) >=3:
            relPos = point - self.pos
            for i in range(len(self.points)-1):
                if orientation(self.points[i],self.points[i+1],relPos) < 0:
                    return False
            if orientation(self.points[-1],self.points[0],relPos) < 0:
                return False
            return True

#Layer classes
class LinearLayer(Layer):
    def __init__(self,dims, parent):
        super().__init__([np.array([0,0]),np.array([32,0]), np.array([32,32]), np.array([0,32])], QColor(132,0,0), np.array([-18,10]), np.array([38,10]), parent, "Linear", dims)
        self.dims = dims
        self.func = nn.Linear(dims[0],dims[1])
        
class RELULayer(Layer):
    def __init__(self, parent):
        super().__init__([np.array([0,0]),np.array([28,16]), np.array([0,32])], QColor(255,201,14), np.array([-18,10]), np.array([34,10]), parent, "RELU", "")
        self.func = nn.ReLU()

class FlattenLayer(Layer):
    def __init__(self,params, parent):
        super().__init__([np.array([0,0]),np.array([32,32])], QColor(127,127,127), np.array([-18,10]), np.array([38,10]), parent, "Flatten", "")
        self.func = nn.Flatten(start_dim=0)

#Deprecated for now
#No params
class NormalizeLayer(Layer):
    def __init__(self, params, parent):
        super().__init__([np.array([0,0]),np.array([32,32])], QColor(127,127,127), np.array([-18,10]), np.array([38,10]), parent)
        self.func = lambda y : (y/255-0.1307)/0.3081

#params = [scale, shift]
class AffineLayer(Layer):
    def __init__(self, params, parent):
        super().__init__([np.array([0,0]),np.array([32,32])], QColor(127,127,127), np.array([-18,10]), np.array([38,10]), parent, "Affine", params)
        self.func = lambda y : y*float(params[0]) + float(params[1])

#No params
class SoftMaxLayer(Layer):
    def __init__(self, params, parent):
        super().__init__([np.array([0,0]),np.array([32,32])], QColor(127,127,127), np.array([-18,10]), np.array([38,10]), parent, "SoftMax", "")
        self.func = nn.Softmax()

#params = Bool
class ImageProcessLayer(Layer):
    def __init__(self, params, parent):
        super().__init__([np.array([0,0]),np.array([64,0]),np.array([64,40]),np.array([0,40])], QColor(0,162,232), None, np.array([70,14]), parent, "ImageToTensor", "[Greyscale]" if params else "")
        if params == True:
            self.func = tt.Grayscale()
        #should always normalize data
        else:
            self.func = lambda y : y      

#Network class
class NeuralNetwork(nn.Module):
    def __init__(self,parent = None):
        self.parent = parent
        super(NeuralNetwork, self).__init__()
        #Ordered!
        self.layers = []
        self.nnlayers = nn.ModuleList()
        self.optimizer = None
    
    def add_layer(self,layer):
        self.layers.append(layer)
        ##
        try:
            self.nnlayers.append(layer.func)
        except:
            pass

    def forward(self,x):
        for layer in self.layers:
            x = layer.func(x)
        return x
    
    #absorb another neural network
    def fuse(self,network,side):
        #add to right side of self
        if side == "right":
            for layer in network.layers:
                self.add_layer(layer)
                layer.parent = self
        #add to left side of self
        elif side == "left":
            for layer in network.layers:
                layer.parent = self
            self.layers = network.layers + self.layers
            self.nnlayers = network.nnlayers + self.nnlayers

    def splice(self, pos):
        self.parent.objectStack.append(NeuralNetwork(self.parent))
        for i in range(len(self.layers)):
            if i == pos:
                self.layers[i].right.activated = False
            elif i > pos:
                self.layers[i].updatePos(self.layers[i].pos + np.array([24,0]))
                if i == pos + 1:
                    self.layers[i].left.activated = False
                self.parent.objectStack[-1].add_layer(self.layers[i])
                self.layers[i].parent = self.parent.objectStack[-1]
        homeLayers = self.layers[:(pos+1)]
        self.layers = []
        self.nnlayers = nn.ModuleList()
        for layer in homeLayers:
            self.add_layer(layer)

    ##debug ver1 here, this
    def Train(self,data,labels):
        self.train()
        for batch_idx in range(len(data)):
            self.optimizer.zero_grad()
            output = self.forward(data[batch_idx])
            loss = tf.nll_loss(output, labels[batch_idx])
            loss.backward()
            self.optimizer.step()

class TransformsToolbar():
    def __init__(self,parent):  
        self.OptionalWidgets = []
        self.parent = parent     
        self.TRANSFORMS = {"Flatten": FlattenLayer,"Normalize": NormalizeLayer, "Softmax": SoftMaxLayer, "Affine": AffineLayer}
        self.transform_toolbar = QToolBar("Transformations toolbar")
        self.transform_toolbar.toggleViewAction().setVisible(False)
        self.transform_toolbar.setMovable(False)
        self.transform_toolbar.setFixedWidth(200)
        self.transform_toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        self.transform_toolbar_container = QWidget()
        self.transform_toolbar_container.layout = QGridLayout()
        self.parent.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.transform_toolbar)

        self.transform_dropdown = QComboBox()
        self.transform_dropdown.view().pressed.connect(self.handleItemPressed)
        self.transform_dropdown.addItem("Flatten")
        self.transform_dropdown.addItem("Normalize")
        self.transform_dropdown.addItem("Softmax")
        self.transform_dropdown.addItem("Affine")
        self.transform_toolbar_container.layout.addWidget(self.transform_dropdown,0,0,1,2)

        transform_button = QPushButton("Add", self.parent)
        transform_button.clicked.connect(self.onTransformButtonClick)
        self.transform_toolbar_container.layout.addWidget(transform_button,3,0,1,2)

        self.transform_toolbar_container.setLayout(self.transform_toolbar_container.layout)
        self.transform_toolbar.addWidget(self.transform_toolbar_container)
        self.transform_toolbar.addSeparator()

        self.transform_toolbar.close()

    ##Fish out params from optional widgets
    def onTransformButtonClick(self,s):
        if self.transform_dropdown.currentText() == "Affine":
            params = [float(self.scale_input.text()),float(self.shift_input.text())]
        else:
            params = []
        self.parent.objectStack.append(NeuralNetwork(self.parent))
        self.parent.objectStack[-1].add_layer(self.TRANSFORMS[self.transform_dropdown.currentText()](params,self.parent.objectStack[-1]))
        self.parent.redraw()

    #cycle through different parameter input options depending on chosen layer type
    def handleItemPressed(self, index):
        for widget in self.OptionalWidgets:
            widget.deleteLater()
        self.OptionalWidgets = []
        if self.transform_dropdown.model().itemFromIndex(index).text() == "Affine":
            scale_label = QLabel("Scale")
            scale_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.transform_toolbar_container.layout.addWidget(scale_label,1,0)

            self.scale_input = QLineEdit(self.transform_toolbar)
            self.scale_input.setFixedWidth(80)
            self.scale_input.setValidator(QDoubleValidator())
            self.transform_toolbar_container.layout.addWidget(self.scale_input,1,1)

            self.OptionalWidgets.append(scale_label)
            self.OptionalWidgets.append(self.scale_input)

            shift_label = QLabel("Shift")
            shift_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.transform_toolbar_container.layout.addWidget(shift_label,2,0)

            self.shift_input = QLineEdit(self.transform_toolbar)
            self.shift_input.setFixedWidth(80)
            self.shift_input.setValidator(QDoubleValidator())
            self.transform_toolbar_container.layout.addWidget(self.shift_input,2,1)
            
            self.OptionalWidgets.append(shift_label)
            self.OptionalWidgets.append(self.shift_input)


class LinearToolbar():
    def __init__(self,parent):  
        self.OptionalWidgets = []
        self.parent = parent     
        self.linear_toolbar = QToolBar("Linear layers toolbar")
        self.linear_toolbar.toggleViewAction().setVisible(False)
        self.linear_toolbar.setMovable(False)
        self.linear_toolbar.setFixedWidth(200)
        self.linear_toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        linear_toolbar_container = QWidget()
        linear_toolbar_container.layout = QGridLayout()
        self.parent.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.linear_toolbar)

        linear_toolbar_label = QLabel("Input dimension")
        linear_toolbar_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        linear_toolbar_container.layout.addWidget(linear_toolbar_label,0,0)

        self.linear_toolbar_input = QLineEdit(self.linear_toolbar)
        self.linear_toolbar_input.setFixedWidth(80)
        self.linear_toolbar_input.setValidator(QIntValidator())
        linear_toolbar_container.layout.addWidget(self.linear_toolbar_input,0,1)

        linear_toolbar_label2 = QLabel("Output dimension")
        linear_toolbar_label2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        linear_toolbar_container.layout.addWidget(linear_toolbar_label2,1,0)

        self.linear_toolbar_input2 = QLineEdit(self.linear_toolbar)
        self.linear_toolbar_input2.setFixedWidth(80)
        self.linear_toolbar_input2.setValidator(QIntValidator())
        linear_toolbar_container.layout.addWidget(self.linear_toolbar_input2,1,1)

        neural_net_button = QPushButton("Add", self.parent)
        neural_net_button.clicked.connect(self.onNeuralButtonClick)
        linear_toolbar_container.layout.addWidget(neural_net_button,2,0,1,2)

        linear_toolbar_container.setLayout(linear_toolbar_container.layout)
        self.linear_toolbar.addWidget(linear_toolbar_container)
        self.linear_toolbar.addSeparator()

        self.linear_toolbar.close()

    ##Fish out params from optional widgets
    def onNeuralButtonClick(self,s):
        if self.linear_toolbar_input.text() != "" and self.linear_toolbar_input2.text() != "":
            parameters = [int(self.linear_toolbar_input.text()),int(self.linear_toolbar_input2.text())]
        else:
            parameters = [0,0]
        self.parent.objectStack.append(NeuralNetwork(self.parent))
        self.parent.objectStack[-1].add_layer(LinearLayer(parameters,self.parent.objectStack[-1]))
        self.parent.redraw()

#Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        #window configs
        super().__init__()
        #Position of top left of canvas from origin, including scale! (therefore it must be a float)
        self.POS = np.array([0,0])
        self.ZOOM = 0
        self.setWindowTitle("Neuralysis")
        self.setFixedSize(QSize(960, 480))
        self.checkedToolbarButton = []
        self.openToolbar = []
        self.objectStack = []
        #Whatever the mouse cursor in on
        self.aspect = None
        #For tools
        self.MODE = "base"

        #Graphics container ---
        self.container = QLabel(self)
        self.setMouseTracking(True)
        self.container.setMouseTracking(True)
        self.container.setGeometry(0, 0, 960, 480)
        self.canvas = QPixmap(960, 480)
        self.canvas.fill(QColor(0, 0, 0, 0))
        self.container.setPixmap(self.canvas)

        self.mousePressPos = None
        self.mouseMovePos = None
        #index in objectStack
        self.selected = None
        ##can be inferred from the object! 
        #Joint,distance
        self.closest = [None,12]
        #Data from dialogs
        self.parameters = 0

        #Tools toolbar
        right_toolbar = QToolBar("Tools toolbar")
        right_toolbar.toggleViewAction().setVisible(False)
        right_toolbar.setMovable(False)
        right_toolbar.setFixedWidth(32)
        right_toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, right_toolbar)

        self.cut_button = QToolButton()
        self.cut_button.clicked.connect(self.cut_mode)
        self.cut_button.setIcon(QIcon('cut_cursor.png'))

        right_toolbar.addWidget(self.cut_button)

        #Main toolbar ---
        toolbar = QToolBar("Main toolbar")
        toolbar.toggleViewAction().setVisible(False)
        toolbar.setMovable(False)
        toolbar.setFixedWidth(120)
        toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        self.addToolBarBreak(Qt.ToolBarArea.LeftToolBarArea)

        #Main: linear layers toolbar --- 
        Linear = LinearToolbar(self)


        #Main: activation functions toolbar --- 
        self.ACTIVATION_FUNCTIONS = {"ReLU": RELULayer}
        self.activation_toolbar = QToolBar("Activation functions toolbar")
        self.activation_toolbar.toggleViewAction().setVisible(False)
        self.activation_toolbar.setMovable(False)
        self.activation_toolbar.setFixedWidth(200)
        self.activation_toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        activation_toolbar_container = QWidget()
        activation_toolbar_container.layout = QGridLayout()
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.activation_toolbar)

        self.activation_dropdown = QComboBox()
        self.activation_dropdown.addItem("ReLU")
        activation_toolbar_container.layout.addWidget(self.activation_dropdown,0,0)

        activation_button = QPushButton("Add", self)
        activation_button.clicked.connect(self.onActivationButtonClick)
        activation_toolbar_container.layout.addWidget(activation_button,1,0)

        activation_toolbar_container.setLayout(activation_toolbar_container.layout)
        self.activation_toolbar.addWidget(activation_toolbar_container)
        self.activation_toolbar.addSeparator()

        self.activation_toolbar.close()

        #Main: transformations toolbar --- 
        Transforms = TransformsToolbar(self)

        #Main: processing layers toolbar --- 
        self.processing_toolbar = QToolBar("Processing layers toolbar")
        self.processing_toolbar.toggleViewAction().setVisible(False)
        self.processing_toolbar.setMovable(False)
        self.processing_toolbar.setFixedWidth(200)
        self.processing_toolbar.setStyleSheet("background-color: rgb(240, 240, 240)")
        processing_toolbar_container = QWidget()
        processing_toolbar_container.layout = QGridLayout()
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.processing_toolbar)

        processing_toolbar_label = QLabel("Greyscale")
        processing_toolbar_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        processing_toolbar_container.layout.addWidget(processing_toolbar_label,0,0)

        self.processing_toolbar_greyscale = QCheckBox(self.processing_toolbar)
        processing_toolbar_container.layout.addWidget(self.processing_toolbar_greyscale,0,1)

        processing_button = QPushButton("Add", self)
        processing_button.clicked.connect(self.onImageProcessButtonClick)
        processing_toolbar_container.layout.addWidget(processing_button,1,0,1,2)

        processing_toolbar_container.setLayout(processing_toolbar_container.layout)
        self.processing_toolbar.addWidget(processing_toolbar_container)
        self.processing_toolbar.addSeparator()

        self.processing_toolbar.close()

        #Main toolbar buttons ---
        self.linear_button = QAction("Linear Layers", self)
        self.linear_button.setCheckable(True)
        self.linear_button.triggered.connect(lambda s: self.onMainToolbarButtonClick(s,self.linear_button,Linear.linear_toolbar))
        toolbar.addAction(self.linear_button)
        button = toolbar.widgetForAction(self.linear_button)
        button.setFixedSize(116, 24)

        toolbar.addSeparator()

        self.activation_button = QAction("Activation Functions", self)
        self.activation_button.setCheckable(True)
        self.activation_button.triggered.connect(lambda s: self.onMainToolbarButtonClick(s,self.activation_button,self.activation_toolbar))
        toolbar.addAction(self.activation_button)
        button = toolbar.widgetForAction(self.activation_button)
        button.setFixedSize(116, 24)

        toolbar.addSeparator()

        self.transform_button = QAction("Transformations", self)
        self.transform_button.setCheckable(True)
        self.transform_button.triggered.connect(lambda s: self.onMainToolbarButtonClick(s,self.transform_button,Transforms.transform_toolbar))
        toolbar.addAction(self.transform_button)
        button = toolbar.widgetForAction(self.transform_button)
        button.setFixedSize(116, 24)

        toolbar.addSeparator()

        self.processing_button = QAction("Processing Layers", self)
        self.processing_button.setCheckable(True)
        self.processing_button.triggered.connect(lambda s: self.onMainToolbarButtonClick(s,self.processing_button,self.processing_toolbar))
        toolbar.addAction(self.processing_button)
        button = toolbar.widgetForAction(self.processing_button)
        button.setFixedSize(116, 24)

        toolbar.addSeparator()

        #Menu ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        new_button = QAction("New Project", self)
        new_button.triggered.connect(self.onNewButtonClick)

        open_button = QAction("Open", self)
        open_button.triggered.connect(self.onOpenButtonClick)

        save_button = QAction("Save", self)
        save_button.triggered.connect(self.onSaveButtonClick)

        file_menu.addAction(new_button)
        file_menu.addAction(open_button)
        file_menu.addAction(save_button)

        edit_menu = menu.addMenu("&Edit")

        undo_button = QAction("Undo", self)
        undo_button.triggered.connect(self.onUndoButtonClick)

        clear_button = QAction("Clear Weights", self)
        clear_button.triggered.connect(self.onClearButtonClick)

        edit_menu.addAction(clear_button)
        edit_menu.addAction(undo_button)

        file_menu = menu.addMenu("&Help")

    #Convert screen coordinates to global coordinates at zoom level 0, on canvas
    def screenToGlobal(self,pos):
        pos = pos - np.array([120,24])
        return (pos + self.POS)*np.power(1.2,-self.ZOOM)

    #Menu buttons ---
    def onFileButtonClick(self, s):
        pass
       
    def onEditButtonClick(self, s):
        pass

    #Menu:File buttons ---
    def onNewButtonClick(self, s):
        self.objectStack = []
        self.redraw()

    def onOpenButtonClick(self, s):
        pass
    
    def onSaveButtonClick(self, s):
        pass

    #Menu:Edit buttons ---
    def onUndoButtonClick(self, s):
        pass
    
    def onClearButtonClick(self, s):
        pass

    #Main toolbar buttons ---
    def onMainToolbarButtonClick(self, s, button,subToolbar):
        if self.openToolbar == []:
            subToolbar.show()
            self.openToolbar = [subToolbar]
            self.checkedToolbarButton = [button]
        else:
            for cButton in self.checkedToolbarButton:
                if cButton != button:
                    cButton.toggle()
                    self.openToolbar[0].close()
                    self.openToolbar = [subToolbar]
                    self.checkedToolbarButton = [button]
                    subToolbar.show()
                else:
                    subToolbar.close()
                    self.openToolbar = []
                    self.checkedToolbarButton = []

    #Activation functions toolbar buttons ---
    def onActivationButtonClick(self, s):
        self.objectStack.append(NeuralNetwork(self))
        self.objectStack[-1].add_layer(self.ACTIVATION_FUNCTIONS[self.activation_dropdown.currentText()](self.objectStack[-1]))
        self.redraw()

    def onImageProcessButtonClick(self,s):
        self.objectStack.append(NeuralNetwork(self))
        self.objectStack[-1].add_layer(ImageProcessLayer(self.processing_toolbar_greyscale.isChecked(),self.objectStack[-1]))
        self.redraw()

    #Mouse event handlers
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mousePressPos = event.position()
            self.mouseMovePos = event.position()
            if self.MODE == "base":
                    for i in range(len(self.objectStack)):
                        for layer in self.objectStack[i].layers:
                            if layer.isInside(self.screenToGlobal(np.array([self.mousePressPos.toPoint().x(),self.mousePressPos.toPoint().y()]))):
                                self.selected = i
                    if self.selected != None :
                        object = self.objectStack.pop(self.selected)
                        self.objectStack.append(object)
                        self.selected = len(self.objectStack)-1
            elif self.MODE == "cut":
                to_cut = []
                for i in range(len(self.objectStack)):
                    for j in range(len(self.objectStack[i].layers)):
                        layer = self.objectStack[i].layers[j]
                        if type(layer.left) != type(None):
                            if layer.left.isInside(self.screenToGlobal(np.array([self.mousePressPos.toPoint().x(),self.mousePressPos.toPoint().y()]))) and layer.left.activated == True:
                                to_cut.append([i,j])
                for pair in to_cut:
                    self.objectStack[pair[0]].splice(pair[1]-1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.MODE = "base"
            QApplication.restoreOverrideCursor()


    def mouseMoveEvent(self, event):
        #only left button to be clicked when dragging
        if event.buttons() == Qt.MouseButton.LeftButton:
            #Check what is clicked
            if self.mousePressPos:
                if self.selected != None:
                    #Find local spheres for snap - inefficient?
                    #object,distance,side
                    if abs(event.position().x()-540) > 419 or abs(event.position().y()-251) > 228:
                        QCursor.setPos(QWidget.mapToGlobal(self,self.mouseMovePos).toPoint())
                    else:
                        self.closest = [None,12]
                        for object in self.objectStack:
                            if object != self.objectStack[self.selected]:
                                #try-except is for layers with no left or right joints. Later need to improve
                                try:
                                    leftDiffRight = np.sqrt(np.dot(self.objectStack[self.selected].layers[0].left.pos - object.layers[-1].right.pos,self.objectStack[self.selected].layers[0].left.pos - object.layers[-1].right.pos))
                                    if leftDiffRight <= min(12,self.closest[1]):
                                        if self.objectStack[self.selected].layers[0].left.activated == False and object.layers[-1].right.activated == False:
                                            self.closest = [object.layers[-1].right,leftDiffRight]
                                except:
                                    pass
                                if self.closest[0] == None:
                                    try:
                                        rightDiffLeft = np.sqrt(np.dot(self.objectStack[self.selected].layers[-1].right.pos - object.layers[0].left.pos,self.objectStack[self.selected].layers[-1].right.pos - object.layers[0].left.pos))
                                        if rightDiffLeft <= min(12,self.closest[1]):
                                            if self.objectStack[self.selected].layers[-1].right.activated == False and object.layers[0].left.activated == False:
                                                self.closest = [object.layers[0].left,rightDiffLeft]
                                    except:
                                        pass
                        delta = (event.position() - self.mouseMovePos).toPoint()
                        for layer in self.objectStack[self.selected].layers:
                            layer.updatePos(layer.pos + np.array([delta.x(),delta.y()])*np.power(1.2,-self.ZOOM))
                        self.redraw()
                        self.mouseMovePos = event.position()
                else:
                    delta = (event.position() - self.mouseMovePos).toPoint()
                    self.POS = self.POS - np.array([delta.x(),delta.y()])
                    self.redraw()
                    self.mouseMovePos = event.position()
        else:
            if self.aspect != None:
                self.aspect.mouseOn = False
            self.aspect = None
            for i in range(len(self.objectStack)):
                for layer in self.objectStack[i].layers:
                    screenPos = event.position().toPoint()
                    if layer.isInside(self.screenToGlobal(np.array([screenPos.x(),screenPos.y()]))):
                        self.aspect = layer
            if self.aspect != None:
                self.aspect.mouseOn = True
            ##can be made more efficient 
            self.redraw()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.closest[0] != None:
                if self.closest[0].side == "right":
                    diff = self.closest[0].pos - self.objectStack[self.selected].layers[0].left.pos
                    self.closest[0].activated = True
                    for i in range(len(self.objectStack[self.selected].layers)):
                        self.objectStack[self.selected].layers[i].updatePos(self.objectStack[self.selected].layers[i].pos + diff)
                    self.objectStack[self.selected].layers[0].left.activated = True
                    self.closest[0].parent.parent.fuse(self.objectStack[self.selected],"right")
                elif self.closest[0].side == "left":
                    diff = self.closest[0].pos - self.objectStack[self.selected].layers[-1].right.pos
                    self.closest[0].activated = True
                    for i in range(len(self.objectStack[self.selected].layers)):
                        self.objectStack[self.selected].layers[i].updatePos(self.objectStack[self.selected].layers[i].pos + diff)
                    self.objectStack[self.selected].layers[-1].right.activated = True
                    self.closest[0].parent.parent.fuse(self.objectStack[self.selected],"left")
                self.objectStack.pop()
            self.mousePressPos = None
            self.selected = None
            self.closest = [None,12]
            self.redraw()

    #Shortcuts handler
    def cut_mode(self):
        if self.MODE != "cut":
            cut_cursor = QPixmap("cut_cursor.png")
            cursor = QCursor(cut_cursor, hotX = 4, hotY = 4)
            QApplication.setOverrideCursor(cursor)
            self.MODE = "cut"
        else:
            QApplication.restoreOverrideCursor()
            self.MODE = "base"

    #Canvas handler
    ##do not need to redraw everything? - one canvas per object
    def redraw(self):
        self.canvas.fill(QColor(0, 0, 0, 0))
        self.painter = QPainter(self.canvas)
        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(10)
        self.painter.setFont(font)

        self.pen = QPen()
        self.pen.setWidth(1)

        self.brush = QBrush()
        self.brush.setStyle(Qt.BrushStyle.SolidPattern)
        #save lines of code by setting temporary position for selected
        #streamline snap joints
        for object in self.objectStack:
            if self.selected and self.objectStack[self.selected] == object and self.closest[0] != None:
                if self.closest[0].side == "right":
                    diff = self.closest[0].pos - object.layers[0].left.pos
                    for i in range(len(object.layers)):
                        object.layers[i].draw(object.layers[i].pos + diff)
                elif self.closest[0].side == "left":
                    diff = self.closest[0].pos - object.layers[-1].right.pos
                    for i in range(len(object.layers)):
                        object.layers[i].draw(object.layers[i].pos + diff)
            else:
                for layer in object.layers:
                    layer.draw(layer.pos)
        self.painter.end()
        self.container.setPixmap(self.canvas)

    #Drag and drop functionality
    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        screenPos = event.position().toPoint()
        if event.mimeData().hasImage:
            for object in self.objectStack:
                if object.layers[0].__class__ == ImageProcessLayer and object.layers[0].isInside(self.screenToGlobal(np.array([screenPos.x(),screenPos.y()]))):
                    event.setDropAction(Qt.DropAction.IgnoreAction)
                    file_path = event.mimeData().urls()[0].toLocalFile()
                    filename, file_extension = os.path.splitext(file_path)
                    if file_extension == "":
                        try:
                            data = np.array(idx2numpy.convert_from_file(file_path + "\data"))
                            data.setflags(write = 1)
                            #fun = tt.Normalize((0.1307,), (0.3081,))
                            data = torch.from_numpy(data).to(torch.float32)
                            labels = np.array(idx2numpy.convert_from_file(file_path + "\label"))
                            labels.setflags(write = 1)
                            labels = torch.from_numpy(labels).to(torch.int64)
                            object.optimizer = optim.SGD(object.parameters(), lr=0.01,momentum=0.5)
                            object.Train(data,labels)
                        except:
                            print("Bad Format!")
                    else:
                        photo = Image.open(file_path)
                        ##Remove this and add splice tool + normalize layer
                        transform = tt.Compose([tt.ToTensor()])
                        event.accept()
                        print(object(transform(photo)))
                    #Only drag onto the topmost neural net
                    break
        else:
            event.ignore()
    
    ##Shrink scroll area to "visible screen"
    def wheelEvent(self, event):
        angle = event.angleDelta()
        if abs(event.position().x()-540) > 419 or abs(event.position().y()-251) > 228:
            pass
        else:
            #Trim event position to within canvas
            ##allow float for POS as well
            screenPos = event.position().toPoint()
            canvasPos =  np.array([screenPos.x() - 120,screenPos.y() - 24])
            if angle.y() > 0 and self.ZOOM < 6:
                self.ZOOM += 1
                self.POS = (self.POS + canvasPos/6)*1.2
                self.redraw()
            elif angle.y() < 0 and self.ZOOM > -6:
                self.ZOOM -= 1
                self.POS = (self.POS/1.2 - canvasPos/6)
                self.redraw()

# You need one (and only one) QApplication instance per application.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()