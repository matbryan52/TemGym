from typing import Iterable, TYPE_CHECKING, NamedTuple, Optional, Tuple, Any
import weakref
import copy

from PySide6.QtGui import QVector3D
from PySide6.QtCore import (
    Slot,
    Qt,
    QTimer,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QTreeView,
)
from PySide6.QtGui import (
    QKeyEvent,
    QStandardItemModel,
    QStandardItem,
)
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from deepdiff import DeepDiff, extract, Delta

import numpy as np

from . import shapes as comp_geom
from .config_model import to_model, ParamTuple
from .utils import as_gl_lines, P2R, R2P
from .widgets import GLImageItem, MyDockLabel

if TYPE_CHECKING:
    from .model import Model
    from . import components as comp
    from . import Radians


LABEL_RADIUS = 0.3
Z_ORIENT = -1
RAY_COLOR = (0., 0.8, 0.)
XYZ_SCALING = np.asarray((1, 1, 1.))
LENGTHSCALING = 1
MRAD = 1e-3
UPDATE_RATE = 100
BKG_COLOR_3D = (150, 150, 150, 255)


class GridGeomParams(NamedTuple):
    w: float
    h: float
    cx: float
    cy: float
    rotation: 'Radians'
    z: float
    shape: Optional[Tuple[int, int]]


class GridGeomMixin:
    def _get_extents(self) -> GridGeomParams:
        # (cx, cy, w, h, rotation, z)
        raise NotImplementedError()

    def _get_image(self):
        return np.asarray(
            (((255, 128, 128, 255),),),
            dtype=np.uint8,
        )

    def _get_grid_verts(self):
        geom = self._get_extents()
        rotation = geom.rotation
        shape = geom.shape
        if shape is None or any(s <= 0 for s in shape):
            return None
        vertices = self._get_mesh(rotation=0.)
        min_x, min_y, z = vertices.min(axis=0)
        max_x, max_y, _ = vertices.max(axis=0)
        ny, nx = shape
        # this can be done with clever striding / reshaping
        xvals = np.linspace(min_x, max_x, num=nx + 1, endpoint=True)[1:-1]
        xfill = np.asarray((min_y, max_y))
        xfill = np.tile(xfill, xvals.size)
        xvals = np.repeat(xvals, 2)
        yvals = np.linspace(min_y, max_y, num=ny + 1, endpoint=True)[1:-1]
        yfill = np.asarray((min_x, max_x))
        yfill = np.tile(yfill, yvals.size)
        yvals = np.repeat(yvals, 2)

        if rotation != 0.:
            mag, ang = R2P(xvals + xfill * 1j)
            xcplx = P2R(mag, ang + rotation)
            xvals, xfill = xcplx.real, xcplx.imag
            mag, ang = R2P(yfill + yvals * 1j)
            ycplx = P2R(mag, ang + rotation)
            yfill, yvals = ycplx.real, ycplx.imag

        xlines = np.stack((xvals, xfill, np.full_like(xvals, z)), axis=1)
        ylines = np.stack((yfill, yvals, np.full_like(yvals, z)), axis=1)
        return np.concatenate((xlines, ylines), axis=0)

    def get_geom(self):
        vertices = self._get_mesh()
        vertices *= XYZ_SCALING
        self.geom_border = gl.GLLinePlotItem(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0., 0., 0., 8.),
            antialias=True,
            mode='line_strip',
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            grid_verts *= XYZ_SCALING
            self.geom_grid = gl.GLLinePlotItem(
                pos=grid_verts,
                color=(0., 0., 0., 0.2),
                antialias=True,
                mode='lines',
            )
        self.geom_image = GLImageItem(
            vertices,
            self._get_image(),
        )
        return [self.geom_image, self.geom_grid, self.geom_border]

    def update_geometry(self):
        vertices = self._get_mesh()
        vertices *= XYZ_SCALING
        self.geom_image.setVertices(
            vertices,
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            grid_verts *= XYZ_SCALING
            self.geom_grid.setData(
                pos=grid_verts,
                color=(0., 0., 0., 0.3),
                antialias=True,
            )
        self.geom_border.setData(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0., 0., 0., 1.),
            antialias=True,
        )

    def _get_mesh(self, rotation=None):
        geom = self._get_extents()
        if rotation is None:
            rotation = geom.rotation
        vertices, _ = comp_geom.rectangle(
            w=geom.w,
            h=geom.h,
            x=geom.cx,
            y=geom.cy,
            z=Z_ORIENT * geom.z,
            rotation=rotation,
        )
        return vertices


class ComponentGUIWrapper:
    def __init__(self, component: 'comp.Component', window: 'TemGymWindow'):
        self.component = component
        self.window = weakref.ref(window)

    def get_parameters(self) -> dict[str, Any]:
        return self.component.get_parameters()

    def update_geometry(self):
        pass

    def get_label(self) -> gl.GLTextItem:
        return gl.GLTextItem(
            pos=np.array([
                -LABEL_RADIUS,
                LABEL_RADIUS,
                Z_ORIENT * self.component.z
            ]),
            text=self.component.name,
            color='w',
        )

    def get_geom(self) -> Iterable[gl.GLLinePlotItem]:
        raise NotImplementedError()


class TemGymWindow(QMainWindow):
    '''
    Create the UI Window
    '''
    def __init__(self):

        '''Init important parameters

        Parameters
        ----------
        model : class
            Microscope model
        '''
        super().__init__()

        # Set some main window's properties
        self.setWindowTitle("TemGymBasic")
        self.resize(1600, 950)

        # Create Docks
        label = MyDockLabel("3D View")
        self.tem_dock = Dock("3D View", size=(4, 10), label=label)
        label = MyDockLabel("Detector")
        self.detector_dock = Dock("Detector", size=(4, 6), label=label)
        label = MyDockLabel("Controls")
        self.gui_dock = Dock("Parameters", size=(3.5, 10), label=label)

        self.params_tree = QTreeView()
        self.params_tree.setAlternatingRowColors(True)
        self.params_tree.setSortingEnabled(False)
        self.params_tree.setHeaderHidden(False)
        self.params_tree.setIndentation(10)
        self._current_config = {}
        self.gui_dock.addWidget(self.params_tree)
        self.centralWidget = DockArea()
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.addDock(self.tem_dock, "left")
        self.centralWidget.addDock(self.gui_dock, "right")
        self.centralWidget.addDock(self.detector_dock, "bottom", self.gui_dock)

        # Create the display and the buttons
        self.create3DDisplay()
        self.createDetectorDisplay()

        self._wobble_timer = QTimer(parent=self)
        self._wobble_var = 0.
        self._wobble_params = []
        self._wobble_timer.timeout.connect(self._wobble_cb)

    def set_model(
        self,
        model: 'Model',
        tree: bool = True,
        geometry: bool = True,
        camera: bool = True
    ):
        gui_components = tuple(
            c.gui_wrapper()(c, self) for c in model.components
        )
        if geometry:
            self.add_geometry(gui_components)
        if camera:
            self.update_camera(gui_components)
        if tree:
            self.update_tree(gui_components)
        num_rays = 64
        self.update_rays(model, num_rays)

    @staticmethod
    def qtmodel_to_dict(model: QStandardItemModel):
        params = []
        for i in range(model.rowCount()):
            component = model.item(i)
            params.append({"type": component.text()})
            for j in range(component.rowCount()):
                key = component.child(j, 0)
                value = component.child(j, 1)
                try:
                    py_val = eval(value.text(), {})
                except (NameError, SyntaxError):
                    py_val = value.text()
                params[-1][key.text()] = ParamTuple(
                    py_val,
                    {
                        'checked': key.checkState().value == 2,
                    }
                )
        return params

    @Slot()
    def handleChanged(self, item: QStandardItem):
        params = self.qtmodel_to_dict(item.model())
        diff = DeepDiff(params, self._current_config)
        self._current_config = params
        param_change = False
        for path, _ in diff.get("values_changed", {}).items():
            if path.endswith("['checked']"):
                base_path = path.replace(".meta['checked']", "")
                if item.checkState().value == 2:
                    self._wobble_params.append(base_path)
                else:
                    self._wobble_params = [k for k in self._wobble_params if k != base_path]
                continue
            param_change = True
        if param_change:
            new_model = to_model({"components": params})
            self.set_model(new_model)
        wobbling = self._wobble_timer.isActive()
        if self._wobble_params and not wobbling:
            self._wobble_timer.start(UPDATE_RATE)
        elif not self._wobble_params and wobbling:
            self._wobble_timer.stop()

    def _wobble_cb(self):
        if not self._wobble_params:
            self._wobble_timer.stop()
            return
        self._wobble_var += np.pi / 8
        self._wobble_var = np.angle(np.exp(1j * self._wobble_var))
        param_mult = np.sin(self._wobble_var) * 0.1

        temp_config = copy.deepcopy(self._current_config)
        delta = {"values_changed": {}}
        for wobble_path in self._wobble_params:
            val_path = wobble_path + ".val"
            val = extract(temp_config, val_path)
            new_val = val + (val * param_mult)
            delta["values_changed"][val_path] = {
                "new_value": new_val,
                "old_value": val,
            }
        delta = Delta(delta)
        temp_config = temp_config + delta
        new_model = to_model({"components": temp_config})
        self.set_model(
            new_model,
            tree=False,
            camera=False,
        )

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()

    def create3DDisplay(self):
        '''Create the 3D Display
        '''
        # Create the 3D TEM Widnow, and plot the components in 3D
        self.tem_window = gl.GLViewWidget()
        self.tem_window.setBackgroundColor(BKG_COLOR_3D)

        # Get the model mean height to centre the camera origin
        mean_z = 0.
        mean_z *= Z_ORIENT

        xyoffset = (0.2 * mean_z, -0.2 * mean_z)
        # Define Camera Parameters
        initial_camera_params = {
            'center': QVector3D(*xyoffset, mean_z),
            'fov': 35,
            'azimuth': 45.0,
            'distance': 3.5 * abs(mean_z),
            'elevation': 25.0,
        }
        self.tem_window.setCameraParams(**initial_camera_params)

        self.ray_geometry = gl.GLLinePlotItem(
            mode='lines',
            width=2
        )

        # Add the window to the dock
        self.tem_dock.addWidget(self.tem_window)

    def createDetectorDisplay(self):
        '''Create the detector display
        '''
        # Create the detector window, which shows where rays land at the bottom
        self.detector_window = pg.GraphicsLayoutWidget()
        self.detector_window.setBackground(BKG_COLOR_3D)
        self.detector_window.setAspectLocked(1.0)
        self.spot_img = pg.ImageItem(border=None)
        v2 = self.detector_window.addViewBox()
        v2.setAspectLocked()

        # Invert coordinate system so spot moves up when it should
        v2.invertY()
        v2.addItem(self.spot_img)

        self.detector_dock.addWidget(self.detector_window)

    @Slot()
    def update_rays(self, model: 'Model', num_rays: int):
        try:
            all_rays = tuple(model.run_iter(
                num_rays,
            ))
        except IndexError:
            return

        vertices = as_gl_lines(all_rays, z_mult=Z_ORIENT)
        self.ray_geometry.setData(
            pos=vertices * XYZ_SCALING,
            color=RAY_COLOR + (0.05,),
        )

        image = model.detector.get_image(
            all_rays,
            interfere=False,
        )
        self.spot_img.setImage(np.abs(image.T) ** 2)

        if model.detector is not None:
            pass

    def update_tree(self, components: list[ComponentGUIWrapper]):
        params_model = QStandardItemModel()
        params_model.setHorizontalHeaderLabels(['Parameter', 'Value'])
        params_model.itemChanged.connect(self.handleChanged)
        for gui_comp in components:
            comp_row = QStandardItem(type(gui_comp.component).__name__)
            comp_row.setEditable(False)
            for name, val in gui_comp.get_parameters().items():
                key = QStandardItem(name)
                key.setCheckable(True)
                key.setEditable(False)
                comp_row.appendRow([
                    key,
                    QStandardItem(f"{val}"),
                ])
            params_model.appendRow(comp_row)
        self._current_config = self.qtmodel_to_dict(params_model)
        self.params_tree.setModel(params_model)
        self.params_tree.expandAll()

    def add_geometry(self, components: list[ComponentGUIWrapper]):
        self.tem_window.clear()
        # Loop through all of the model components
        # and add their geometry to the TEM window.
        # FIXME Add in reverse to simulate better depth stacking
        for component in reversed(components):
            for geometry in component.get_geom():
                self.tem_window.addItem(geometry)
        # Add labels next so they appear above geometry
        for component in reversed(components):
            label = component.get_label()
            self.tem_window.addItem(label)
        # Add the ray geometry last so it is always on top
        self.tem_window.addItem(self.ray_geometry)

    def update_camera(self, components: list[ComponentGUIWrapper]):
        z_vals = tuple(c.component.z for c in components)
        mid_z = (min(z_vals) + max(z_vals)) / 2.
        mid_z *= Z_ORIENT
        xyoffset = (0.2 * mid_z, -0.2 * mid_z)
        self.tem_window.setCameraParams(center=QVector3D(*xyoffset, mid_z))

    @Slot()
    def update_geometry(self):
        return
        for component in self.gui_components:
            component.update_geometry()


class ModelComponent(NamedTuple):
    name: str = "Model"


class BasicLensGUI(ComponentGUIWrapper):
    @property
    def lens(self) -> 'comp.Lens':
        return self.component

    def get_geom(self):
        vertices = comp_geom.lens(
            0.2,
            Z_ORIENT * self.component.z,
            64,
        )
        f = Z_ORIENT * self.lens.f
        z = Z_ORIENT * self.lens.z
        dot_size = 10  # Adjust the size of the dot as needed
        f_color = (0, 1, 0, 1)
        self.f_dot = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, z + f]]),
            size=dot_size,
            color=f_color,
        )
        self.f_text = gl.GLTextItem(
            pos=np.array([0, 0, z + f]),
            text=f"f {self.component.name}",
            color='w',
        )
        self.geom = [
            gl.GLLinePlotItem(
                pos=vertices.T,
                color="white",
                width=5
            ),
            self.f_dot,
            self.f_text,
        ]
        return self.geom

    def update_geometry(self):
        z = Z_ORIENT * self.lens.z
        f = Z_ORIENT * self.lens.f

        f_dot = self.geom[-2]
        f_text = self.geom[-1]

        f_dot.setData(pos=np.array([[0, 0, z + f]]))
        f_text.setData(pos=np.array([0, 0, z + f]))


class SourceGUI(ComponentGUIWrapper):
    @property
    def beam(self) -> 'comp.Source':
        return self.component

    def _get_geom(self):
        raise NotImplementedError("Needs to be defined")

    def get_geom(self):
        self.geom = gl.GLLinePlotItem(
            pos=self._get_geom() * XYZ_SCALING,
            color=RAY_COLOR + (0.9,),
            width=2,
            antialias=True,
        )
        return [self.geom]

    def update_geometry(self):
        self.geom.setData(
            pos=self._get_geom() * XYZ_SCALING,
            color=RAY_COLOR + (0.9,),
            antialias=True,
        )


class ParallelBeamGUI(SourceGUI):
    def _get_geom(self):
        return comp_geom.lens(
            self.beam.radius,
            Z_ORIENT * self.component.z,
            64,
            cxy=self.beam.centre_yx[::-1]
        ).T


class GaussBeamGUI(ParallelBeamGUI):
    ...


class PointBeamGUI(SourceGUI):
    def _get_geom(self):
        return comp_geom.lens(
            0.,
            Z_ORIENT * self.component.z,
            64,
            cxy=self.beam.centre_yx[::-1]
        ).T


class SampleGUI(GridGeomMixin, ComponentGUIWrapper):
    @property
    def sample(self) -> 'comp.Sample':
        return self.component

    def _get_image(self):
        return np.asarray(
            (((0, 128, 255, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy, sx = self.sample.shape
        py = self.sample.pixel_size
        px = self.sample.pixel_size
        return GridGeomParams(
            cx=-1 * px / 2.,
            cy=-1 * py / 2.,
            w=sx * px,
            h=sy * py,
            rotation=self.sample.rotation,
            z=self.component.z,
            shape=self.sample.shape,
        )


class STEMSampleGUI(SampleGUI):
    @property
    def sample(self) -> 'comp.STEMSample':
        return self.component

    def _get_image(self):
        return np.asarray(
            (((0, 128, 255, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy = self.sample.scan_shape[0]
        sx = self.sample.scan_shape[1]
        py, px = self.sample.scan_step_yx
        return GridGeomParams(
            cx=-1 * px / 2.,
            cy=-1 * py / 2.,
            w=sx * px,
            h=sy * py,
            rotation=self.sample.scan_rotation_rad,
            z=self.component.z,
            shape=self.sample.scan_shape,
        )


class DeflectorGUI(ComponentGUIWrapper):
    @property
    def deflector(self) -> 'comp.Deflector':
        return self.component

    def get_geom(self):
        self.geom = []
        phi = np.pi / 2
        radius = 0.25
        n_arc = 64

        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.entrance_z,
            n_arc=n_arc,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        return self.geom


class DoubleDeflectorGUI(ComponentGUIWrapper):
    @property
    def d_deflector(self) -> 'comp.DoubleDeflector':
        return self.component

    def get_geom(self):
        self.geom = []
        phi = np.pi / 2
        radius = 0.25
        n_arc = 64

        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.entrance_z,
            n_arc=n_arc,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.exit_z,
            n_arc=n_arc,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        return self.geom


class SimpleDoubleDeflectorGUI(DoubleDeflectorGUI):
    ...


class DetectorGUI(GridGeomMixin, ComponentGUIWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_type = 'intensity'

    @property
    def detector(self) -> 'comp.Detector':
        return self.component

    def _get_image(self):
        return np.asarray(
            (((10, 190, 100, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy, sx = self.detector.shape
        pixelsize = self.detector.pixel_size
        # cx, cy, w, h, rotation, z, px_shape
        return GridGeomParams(
            cx=-1 * pixelsize / 2.,
            cy=-1 * pixelsize / 2.,
            w=sx * pixelsize,
            h=sy * pixelsize,
            rotation=self.detector.rotation_rad,
            z=self.component.z,
            shape=self.detector.shape,
        )


class AccumulatingDetectorGUI(DetectorGUI):
    @property
    def detector(self) -> 'comp.AccumulatingDetector':
        return self.component


class ApertureGUI(BasicLensGUI):
    @property
    def aperture(self) -> 'comp.Aperture':
        return self.component


class BiprismGUI(ComponentGUIWrapper):
    @property
    def biprism(self) -> 'comp.Biprism':
        return self.component

    def _get_geom(self):
        return comp_geom.biprism(
            Z_ORIENT*self.biprism.z,
            0.5,
            self.biprism.rotation_rad,
            self.biprism.offset,
        )

    def get_geom(self):
        self.geom = gl.GLLinePlotItem(
            pos=self._get_geom(),
            color='white',
            width=5,
            antialias=True,
        )
        return [self.geom]

    def update_geometry(self):
        self.geom.setData(
            pos=self._get_geom(),
            color='white',
            antialias=True,
        )


class ProjectorLensSystemGUI(ComponentGUIWrapper):
    @property
    def pl_system(self) -> 'comp.ProjectorLensSystem':
        return self.component

    def get_geom(self):
        self.geom = []

        vertices = comp_geom.lens(
            0.2,
            Z_ORIENT * self.component.entrance_z,
            64,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=vertices.T,
                color="white",
                width=5
            )
        )

        vertices = comp_geom.lens(
            0.2,
            Z_ORIENT * self.component.exit_z,
            64,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=vertices.T,
                color="white",
                width=5
            )
        )

        return self.geom
