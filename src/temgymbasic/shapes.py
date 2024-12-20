import triangle as tr
from .utils import P2R, R2P
from . import Radians
import numpy as xp


def rectangle(w, h, x, y, z, rotation: Radians = 0.):
    '''Generates vertices for a square 3D model. Used to represent the detector

    Parameters
    ----------
    w : float
        Width of the square wire model
    x : float
        X position of the square wire model
    y : float
        Wire position of the square wire model
    z : float
        Wire position of the square wire model

    Returns
    -------
    verts3D: ndarray
        vertices to draw a 3D model
    '''

    vertices = xp.array(
        [[x + w / 2, y + h / 2, z],
         [x - w / 2, y + h / 2, z],
         [x - w / 2, y - h / 2, z],
         [x + w / 2, y - h / 2, z]]
    )
    if rotation != 0.:
        mag, ang = R2P(vertices[:, 0] + vertices[:, 1] * 1j)
        _vertices = P2R(mag, ang + rotation)
        vertices[:, 0] = _vertices.real
        vertices[:, 1] = _vertices.imag
    faces = xp.array(
        [[0, 1, 2],
         [0, 2, 3]]
    )
    return vertices, faces


def deflector(r, phi, z, n_arc):
    '''Wire model geometry of deflector

    Parameters
    ----------
    r : float
        Radius of deflector geometry
    phi : float
        Angular width of deflector mode
    z : float
        Z position of deflector geometry
    n_arc : int
        Number of arcs to use to make up the model

    Returns
    -------
    points_arc_1 : ndarray
        Points of a circle to represent the lens geometry
    points_arc_2 : ndarray
        Points of a circle to represent the lens geometry
    '''

    THETA = xp.linspace(-phi, phi, n_arc, endpoint=True)
    R = r*xp.ones(xp.size(THETA))
    Z = z*xp.ones(xp.size(THETA))

    points_arc_1 = xp.array([R*xp.cos(THETA), R*xp.sin(THETA), Z])
    points_arc_2 = xp.array([-R*xp.cos(THETA), -R*xp.sin(-THETA), Z])

    return points_arc_1, points_arc_2


def lens(r, z, n_arc, cxy=(0., 0.)):
    '''Wire model geometry of lens

    Parameters
    ----------
    r : float
        Radius of lens geometry
    z : float
        Z position of lens geometry
    n_arc : int
        Number of arcs to use to make up the model

    Returns
    -------
    points_circle : ndarray
        Points of a circle to represent the lens geometry
    '''
    THETA = xp.linspace(0, 2*xp.pi, n_arc, endpoint=True)
    R = r*xp.ones(xp.size(THETA))
    Z = z*xp.ones(xp.size(THETA))
    cx, cy = cxy

    points_circle = xp.array([R*xp.cos(THETA) + cx, R*xp.sin(THETA) + cy, Z])

    return points_circle


def biprism(z, r, theta: Radians, offset):
    '''Wire model geometry for biprism

    Parameters
    ----------
    z : float
        z position of wire
    r : float
        Radius of wire
    theta : float
        Angle of wire in radians
    offset : float
        offset of wire along x-axis

    Returns
    -------
    points : ndarray
        Points array of wire geometry
    '''
    biprism_loc_v = xp.array([offset * xp.cos(theta), offset * xp.sin(theta)])
    biprism_v = xp.array([-xp.sin(theta), xp.cos(theta)])

    points = xp.array([
            biprism_loc_v - r * biprism_v,
            biprism_loc_v + r * biprism_v,
    ])

    return xp.concatenate((
            points,
            xp.array((z, z)).reshape(2, 1)
        ),
        axis=1,
    )


def quadrupole(r, phi, z, n_arc):
    '''Wire model geometry of deflector

    Parameters
    ----------
    r : float
        Radius of quadrupole geometry
    phi : float
        Angular width of quadrupole mode
    z : float
        Z position of quadrupole geometry
    n_arc : int
        Number of arcs to use to make up the model

    Returns
    -------
    points_arc_1 : ndarray
        Points of first semi circle that represent the quadrupole geometry
    points_arc_2 : ndarray
        Points of second semi circle that represent the quadrupole geometry
    points_arc_3 : ndarray
        Points of third semi circle that represent the quadrupole geometry
    points_arc_4 : ndarray
        Points of fourth semi circle that represent the quadrupole geometry
    '''

    THETA = xp.linspace(-phi, phi, n_arc, endpoint=True)
    R = r*xp.ones(xp.size(THETA))
    Z = z*xp.ones(xp.size(THETA))

    points_arc_1 = xp.array([R*xp.cos(THETA), R*xp.sin(THETA), Z])
    points_arc_2 = xp.array([-R*xp.cos(THETA), -R*xp.sin(-THETA), Z])
    points_arc_3 = xp.array([R*xp.cos(THETA+xp.pi/2), R*xp.sin(THETA+xp.pi/2), Z])
    points_arc_4 = xp.array([-R*xp.cos(THETA+xp.pi/2), -R*xp.sin(-THETA+xp.pi/2), Z])

    return points_arc_1, points_arc_2, points_arc_3, points_arc_4


def aperture(r_i, r_o, n_i, n_o, x, y, z):
    '''3D vertices model of an aperture

    Parameters
    ----------
    r_i : float
        Radius of inner aperture model
    r_o : float
        Radius of outer aperture model
    n_i : int
       Number of points used to represent inner aperture model
    n_o : int
        Number of points used to represent outer aperture model
    x : float
        X position of aperture
    y : float
        Y position of aperture
    z : float
        Z position of aperture

    Returns
    -------
    verts3D : ndarray
        3D array of vertices that represent the aperture model
    '''
    i_i = xp.arange(n_i)
    i_o = xp.arange(n_o)
    theta_i = i_i * 2 * xp.pi / n_i
    theta_o = i_o * 2 * xp.pi / n_o
    pts_inner = xp.stack([x + xp.cos(theta_i)*r_i, y + xp.sin(theta_i)*r_i], axis=1)
    pts_outer = xp.stack([x + xp.cos(theta_o)*r_o, y + xp.sin(theta_o)*r_o], axis=1)
    seg_i = xp.stack([i_i, i_i + 1], axis=1) % n_i
    seg_o = xp.stack([i_o, i_o + 1], axis=1) % n_o

    pts = xp.vstack([pts_outer, pts_inner])
    seg = xp.vstack([seg_o, seg_i + seg_o.shape[0]])

    aperture_dict = dict(vertices=pts, segments=seg, holes=[[x, y]])
    aperture_tri = tr.triangulate(aperture_dict, 'qpa0.05')

    zverts = xp.ones((aperture_tri['triangles'].shape[0], 3, 1), dtype=xp.float32)*z
    verts_2D = aperture_tri['vertices'][aperture_tri['triangles']]
    verts_3D = xp.dstack([verts_2D, zverts])

    return verts_3D
