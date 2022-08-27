import PyQt5
from PyQt5.QtWidgets import (
    QSlider,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QCheckBox
)

import numpy as np
from pyqtgraph.Qt import QtCore
from functools import partial

class LensGui():
    def __init__(self, name, f):
        
        self.box = QGroupBox(name)
        self.fslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fslider.setMinimum(-1000)
        self.fslider.setMaximum(-10)
        self.fslider.setValue(int(round(f*1000)))
        self.fslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.flabel = QLabel('Focal Length = ' + "{:.2f}".format(f))
        self.flabel.setMinimumWidth(80)

        hbox = QHBoxLayout()
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.flabel)
        hbox.addSpacing(10)
        hbox.addWidget(self.fslider)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        vbox.addStretch()

        self.box.setLayout(vbox)
        
class DeflectorGui():
    def __init__(self, name, defx, defy):
        
        self.box = QGroupBox(name)
        self.defxslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.defxslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.defxslider.setMinimum(-3000)
        self.defxslider.setMaximum(3000)
        self.defxslider.setValue(int(round(defx*1000)))
        self.defxslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.defxlabel = QLabel('X Deflection = ' + "{:.2f}".format(defx))
        self.defxlabel.setMinimumWidth(80)
        
        self.defyslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.defyslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.defyslider.setMinimum(-3000)
        self.defyslider.setMaximum(3000)
        self.defyslider.setValue(int(round(defx*1000)))
        self.defyslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.defylabel = QLabel('Y Deflection = ' + "{:.2f}".format(defy))
        self.defylabel.setMinimumWidth(80)
        
        vbox = QVBoxLayout()
        self.def_slider_label = QLabel('Deflector Sliders')
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.def_slider_label)
        hbox_labels.addStretch()
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.defxslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.defxlabel)
        hbox.addSpacing(10)
        hbox.addWidget(self.defyslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.defylabel)
        
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        vbox.addStretch()
    
        self.box.setLayout(vbox)
        
class DoubleDeflectorGui():
    def __init__(self, name, updefx, updefy, lowdefx, lowdefy):
        self.button_wobble = QCheckBox("Wobble Upper Deflector")
        self.box = QGroupBox(name)
        self.updefxslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.updefxslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.updefxslider.setMinimum(-3000)
        self.updefxslider.setMaximum(3000)
        self.updefxslider.setValue(int(round(updefx*1000)))
        self.updefxslider.setTickPosition(QSlider.TicksBelow)
        

        # %%% Create sliders for control of both lenses
        self.updefxlabel = QLabel('X Deflection = ' + "{:.2f}".format(updefx))
        self.updefxlabel .setMinimumWidth(80)
        
        self.updefyslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.updefyslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.updefyslider.setMinimum(-3000)
        self.updefyslider.setMaximum(3000)
        self.updefyslider.setValue(int(round(updefx*1000)))
        self.updefyslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.updefylabel = QLabel('Y Deflection = ' + "{:.2f}".format(updefy))
        self.updefylabel .setMinimumWidth(80)
        
        vbox = QVBoxLayout()
        self.def_slider_label = QLabel('Upper Deflector Sliders')
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.def_slider_label)
        hbox_labels.addStretch()
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.updefxslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.updefxlabel)
        hbox.addSpacing(10)
        hbox.addWidget(self.updefyslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.updefylabel)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        vbox.addSpacing(20)
        
        self.lowdefxslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.lowdefxslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lowdefxslider.setMinimum(-3000)
        self.lowdefxslider.setMaximum(3000)
        self.lowdefxslider.setValue(int(round(lowdefx*1000)))
        self.lowdefxslider.setTickPosition(QSlider.TicksBelow)
        

        # %%% Create sliders for control of both lenses
        self.lowdefxlabel = QLabel('X Deflection = ' + "{:.2f}".format(lowdefx))
        self.lowdefxlabel.setMinimumWidth(80)
        
        self.lowdefyslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.lowdefyslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lowdefyslider.setMinimum(-3000)
        self.lowdefyslider.setMaximum(3000)
        self.lowdefyslider.setValue(int(round(lowdefx*1000)))
        self.lowdefyslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.lowdefylabel = QLabel('Y Deflection = ' + "{:.2f}".format(lowdefy))
        self.lowdefylabel .setMinimumWidth(80)
        
        
        self.def_slider_label = QLabel('Lower Deflector Sliders')
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.def_slider_label)
        hbox_labels.addStretch()
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.lowdefxslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.lowdefxlabel)
        hbox.addSpacing(10)
        hbox.addWidget(self.lowdefyslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.lowdefylabel)
        
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        vbox.addSpacing(20)
        
        self.xbuttonwobble = QCheckBox("Wobble Upper Deflector X")
        self.defratioxlabel = QLabel('Deflector X Response Ratio = ' + "{:.2f}".format(0))
        self.defratioxslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.defratioxslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.defratioxslider.setMinimum(-3000)
        self.defratioxslider.setMaximum(3000)
        self.defratioxslider.setValue(-2000)
        self.defratioxslider.setTickPosition(QSlider.TicksBelow)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.xbuttonwobble)
        hbox.addSpacing(10)
        hbox.addWidget(self.defratioxslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.defratioxlabel)
        
        vbox.addLayout(hbox)

        self.ybuttonwobble = QCheckBox("Wobble Upper Deflector Y")
        self.defratioylabel = QLabel('Deflector Y Response Ratio = ' + "{:.2f}".format(0))
        self.defratioyslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.defratioyslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.defratioyslider.setMinimum(-3000)
        self.defratioyslider.setMaximum(3000)
        self.defratioyslider.setValue(-2000)
        self.defratioyslider.setTickPosition(QSlider.TicksBelow)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.ybuttonwobble)
        hbox.addSpacing(10)
        hbox.addWidget(self.defratioyslider)
        hbox.addSpacing(10)
        hbox.addWidget(self.defratioylabel)
        
        vbox.addLayout(hbox)
        vbox.addStretch()
        
        self.box.setLayout(vbox)
        
class BiprismGui():
    def __init__(self, name, deflection, theta):
        
        self.box = QGroupBox(name)
        self.defslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.defslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.defslider.setMinimum(-9000)
        self.defslider.setMaximum(9000)
        self.defslider.setValue(int(round(deflection*1000)))
        self.defslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.deflabel = QLabel('Biprism Deflection = ' + "{:.2f}".format(deflection))
        self.deflabel.setMinimumWidth(80)
        
        vbox = QVBoxLayout()
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.deflabel)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.defslider)
        
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        
        self.rotslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rotslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rotslider.setMinimum(0)
        self.rotslider.setMaximum(1)
        self.rotslider.setValue(theta)
        self.rotslider.setTickPosition(QSlider.TicksBelow)
        
        self.rotlabel = QLabel('Rotation (Radians) = ' + "{:.2f}".format(theta))
        self.rotlabel.setMinimumWidth(80)
        
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.rotlabel)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.rotslider)
        
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        
        vbox.addStretch()
    
        self.box.setLayout(vbox)
        
class ModelGui():
    def __init__(self, num_rays, beam_type, beam_semi_angle):
        
        self.box = QGroupBox('Model Settings')
        self.rayslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rayslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rayslider.setMinimum(4)
        self.rayslider.setMaximum(15)
        self.rayslider.setValue(int(np.log2(num_rays)))
        self.rayslider.setTickPosition(QSlider.TicksBelow)
        
        self.raylabel= QLabel(str(num_rays))
        self.raylabel.setMinimumWidth(80)
        self.modelraylabel = QLabel('Number of Rays')
        
        vbox = QVBoxLayout()
        vbox.addStretch()
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.rayslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.raylabel)
        
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.modelraylabel)
        hbox_labels.addStretch()
        
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        
        self.beamangleslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.beamangleslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.beamangleslider.setMinimum(0)
        self.beamangleslider.setMaximum(157)
        self.beamangleslider.setValue(int(round(beam_semi_angle, 2)*100))
        self.beamangleslider.setTickPosition(QSlider.TicksBelow)
        
        self.beamanglelabel = QLabel(str(round(beam_semi_angle, 2)))
        self.beamanglelabel.setMinimumWidth(80)
        self.modelbeamanglelabel = QLabel('Axial/Paralell Beam Semi Angle')
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.beamangleslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.beamanglelabel)
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.modelbeamanglelabel)
        hbox_labels.addStretch()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        
        self.beamwidthslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.beamwidthslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.beamwidthslider.setMinimum(0)
        self.beamwidthslider.setMaximum(1000)
        self.beamwidthslider.setValue(0)
        self.beamwidthslider.setTickPosition(QSlider.TicksBelow)
        
        self.beamwidthlabel = QLabel('0')
        self.beamwidthlabel.setMinimumWidth(80)
        self.modelbeamwidthlabel = QLabel('Paralell Beam Width')
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.beamwidthslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.beamwidthlabel)
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.modelbeamwidthlabel)
        hbox_labels.addStretch()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)
        
        # creating check box
        self.checkBoxAxial = QCheckBox("Axial Beam")

        # creating check box
        self.checkBoxPoint = QCheckBox("Point Beam")
    
        # creating check box
        self.checkBoxParalell = QCheckBox("Paralell Beam")
    
        # calling the uncheck method if any check box state is changed
        self.checkBoxParalell.stateChanged.connect(partial(self.uncheck, self.checkBoxParalell))
        self.checkBoxPoint.stateChanged.connect(partial(self.uncheck, self.checkBoxPoint))
        self.checkBoxAxial.stateChanged.connect(partial(self.uncheck, self.checkBoxAxial))

        hbox_check_boxes = QHBoxLayout()
        hbox.addWidget(self.checkBoxAxial)
        hbox.addWidget(self.checkBoxPoint)
        hbox.addWidget(self.checkBoxParalell)
        
        vbox.addLayout(hbox_check_boxes)
        
        self.box.setLayout(vbox)
        
        if beam_type == 'axial':
            self.checkBoxAxial.setChecked(True)
        elif beam_type == 'paralell':
            self.checkBoxParalell.setChecked(True)
        elif beam_type == 'point':
            self.checkBoxPoint.setChecked(True)
            
        
    # uncheck method
    def uncheck(self, btn):

        # checking if state is checked
        if btn.isChecked() == True:
    
            # if first check box is selected
            if btn == self.checkBoxAxial:
    
                # making other check box to uncheck
                self.checkBoxParalell.setChecked(False)
                self.checkBoxPoint.setChecked(False)
    
            # if second check box is selected
            elif btn == self.checkBoxParalell:
    
                # making other check box to uncheck
                self.checkBoxAxial.setChecked(False)
                self.checkBoxPoint.setChecked(False)
    
            # if third check box is selected
            elif btn == self.checkBoxPoint:
    
                # making other check box to uncheck
                self.checkBoxAxial.setChecked(False)
                self.checkBoxParalell.setChecked(False)
                
class ApertureGui():
    def __init__(self, name, min_radius, max_radius, inner_radius):
        
        self.box = QGroupBox(name)
        self.radiusslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.radiusslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radiusslider.setMinimum(int(round(min_radius*1000)))
        self.radiusslider.setMaximum(int(round(max_radius*1000)))
        self.radiusslider.setValue(int(round(inner_radius*1000)))
        self.radiusslider.setTickPosition(QSlider.TicksBelow)
        
        # %%% Create sliders for control of both lenses
        self.radiuslabel = QLabel('Aperture Radius = ' + "{:.2f}".format(inner_radius))
        self.radiuslabel.setMinimumWidth(80)
        
        vbox = QVBoxLayout()
        vbox.addStretch()
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.radiusslider)
        
        hbox_label = QHBoxLayout()
        hbox_label.addWidget(self.radiuslabel)
        
        vbox.addLayout(hbox_label)
        vbox.addLayout(hbox)
        vbox.addStretch()
    
        self.box.setLayout(vbox)



            




        