from typing import Optional

from PySide6 import QtCore
from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import (
    QSlider,
    QLabel,
    QHBoxLayout,
    QLayout,
    QWidget,
    QVBoxLayout,
    QLineEdit,
)
from PySide6.QtGui import (
    QIntValidator,
)


class DoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._maxi = 100
        self._numticks = 20
        self._minf = 0.
        self._maxf = 1.
        self.valueChanged.connect(self.emitDoubleValueChanged)
        super().setMinimum(0)
        super().setMaximum(self._maxi)
        self.setTickInterval()

    @property
    def _rangef(self):
        minf, maxf = self._floatMinMax()
        return maxf - minf

    def _floatMinMax(self):
        return self._minf, self._maxf

    def _toFloat(self, val: int):
        return float((val / self._maxi) * self._rangef)

    def _toInt(self, val: float):
        minf, _ = self._floatMinMax()
        return int((val - minf) / self._rangef * self._maxi)

    def emitDoubleValueChanged(self):
        self.doubleValueChanged.emit(self.value())

    def value(self):
        return self._toFloat(super().value())

    def minimum(self):
        return self._toFloat(super().minimum())

    def maximum(self):
        return self._toFloat(super().maximum())

    def setTickInterval(self, ti: int = 0) -> None:
        super().setTickInterval(self._maxi / (self._numticks - 1))

    def setMinimum(self, value: float):
        value_f = self.value()
        self._minf = value
        self._maxf = max(self._minf + 1e-5, self._maxf)
        self.setValue(value_f)

    def setMaximum(self, value: float):
        value_f = self.value()
        self._maxf = value
        self._minf = min(self._maxf - 1e-5, self._minf)
        self.setValue(value_f)

    # def setSingleStep(self, value: float):
    #     super().setSingleStep(1)

    # def singleStep(self):
    #     return 1

    def setValue(self, value: float):
        super().setValue(self._toInt(value))


class QNumericLabel(QLabel):
    def setPrefix(self, prefix: str):
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        try:
            return self._prefix
        except AttributeError:
            return ''

    @Slot(int)
    @Slot(complex)
    def setText(self, v):
        if isinstance(v, float):
            super().setText(f"{self.prefix}{v:.3f}")
        else:
            super().setText(f"{self.prefix}{v}")


def slider_config(slider: QSlider, value: int, vmin: int, vmax: int):
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setMinimum(vmin)
    slider.setMaximum(vmax)
    slider.setValue(value)
    slider.setTickPosition(QSlider.TicksBelow)


def labelled_slider(
    value: int,
    vmin: int,
    vmax: int,
    name: Optional[str] = None,
    prefix: str = '',
    insert_into: Optional[QLayout] = None,
    spacing: int = 15,
    decimals: int = 0,
):
    if decimals > 0:
        slider = DoubleSlider(QtCore.Qt.Orientation.Horizontal)
    else:
        slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider_config(slider, value, vmin, vmax)
    slider_valuelabel = QNumericLabel(prefix + str(slider.value()))
    slider_valuelabel.setPrefix(prefix)
    slider_valuelabel.setMinimumWidth(80)
    if decimals > 0:
        slider.doubleValueChanged.connect(slider_valuelabel.setText)
    else:
        slider.valueChanged.connect(slider_valuelabel.setText)

    if isinstance(insert_into, QHBoxLayout):
        hbox = insert_into
    else:
        hbox = QHBoxLayout()

    if name is not None:
        slider_namelabel = QLabel(name)
        hbox.addWidget(slider_namelabel)

    hbox.addWidget(slider)
    hbox.addSpacing(spacing)
    hbox.addWidget(slider_valuelabel)

    if insert_into is not None and not isinstance(insert_into, QHBoxLayout):
        insert_into.addLayout(hbox)

    return slider, hbox


class LabelledIntField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText(title)
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value is not None:
            self.lineEdit.setText(str(initial_value))
        layout.addWidget(self.lineEdit)
        layout.addStretch()

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)

    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    def getValue(self, default: int = 1):
        try:
            return int(self.lineEdit.text())
        except ValueError:
            return default

    def insert_into(self, layout):
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        return self
