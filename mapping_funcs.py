'''
silly little functions for mapping projects
(C) tlohde
'''
from contourpy import contour_generator
import cartopy.crs as ccrs
import geopandas as gpd
from itertools import groupby, chain
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import (LinearSegmentedColormap,
                               Normalize,
                               BoundaryNorm,
                               ListedColormap)
from matplotlib.cm import ScalarMappable
import numpy as np
import pystac_client
from pystac.extensions.eo import EOExtension as eo
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
import pyproj
import planetary_computer
from rasterio.enums import Resampling
import re
import rioxarray as rio
from scipy.interpolate import griddata
import shapely
from shapely import LineString
from shapelysmooth import taubin_smooth
from skimage import color
import stackstac
from tqdm import tqdm
from typing import Literal
import xarray as xr
import xrspatial as xrs
from xrspatial.classify import natural_breaks, equal_interval, reclassify


def validate_type(func, locals):
    '''
    validate inputs to function
    '''
    for var, var_type in func.__annotations__.items():
        if var == 'return':
            continue
        if not any([isinstance(locals[var], vt) for vt in [var_type]]):
            raise TypeError(
                f'{var} must be (/be one of): {var_type} not a {locals[var]}'
                )


class Utils():
    '''
    general utilities for working with raster
    and vector datasets
    '''

    def twoD_interp(img: np.ndarray) -> np.ndarray:
        '''
        2d interpolation. useful for filling in gaps in
        digital elevation models
        input - 2d numpy arr with gaps as `np.nan`
        output - filled 2d numpy arr
        '''
        assert len(img.shape) == 2, 'input must be 2d array'
        validate_type(Utils.twoD_interp, locals=locals())
        h, w = img.shape[:2]
        mask = np.isnan(img)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        known_x = xx[~mask]
        known_y = yy[~mask]
        known_z = img[~mask]
        missing_x = xx[mask]
        missing_y = yy[mask]

        interp_vals = griddata((known_x, known_y),
                               known_z, (missing_x, missing_y),
                               method='cubic', fill_value=np.nan)
        interpolated = img.copy()
        interpolated[missing_y, missing_x] = interp_vals
        return interpolated

    def get_local_utm(geom: shapely.geometry):
        '''
        get epsg code for utm zone input geometry is in
        assumes shapely geometry is lat/lon epsg4326 coords
        '''
        _minx, _miny, _maxx, _maxy = geom.bounds
        _utms = query_utm_crs_info(
            'WGS84',
            area_of_interest=AreaOfInterest(
                west_lon_degree=_minx,
                south_lat_degree=_miny,
                east_lon_degree=_maxx,
                north_lat_degree=_maxy
            )
        )
        return ccrs.epsg(_utms[0].code)
    
    def shapely_reprojector(geo: shapely.geometry,
                            src_crs: int=3413,
                            target_crs: int=4326):
        
        """
        reproject shapely point (geo) from src_crs to target_crs
        avoids having to create geopandas series to handle crs transformations
        """

        assert isinstance(geo,
                          (shapely.geometry.polygon.Polygon,
                           shapely.geometry.linestring.LineString,
                           shapely.geometry.point.Point)
                          ), 'geo must be shapely geometry'
        
        transformer = pyproj.Transformer.from_crs(
            src_crs,
            target_crs,
            always_xy=True
        )
        
        if isinstance(geo, shapely.geometry.point.Point):
            _x, _y = geo.coords.xy
            return shapely.Point(*transformer.transform(_x, _y))
        elif isinstance(geo, shapely.geometry.linestring.LineString):
            _x, _y = geo.coords.xy
            return shapely.LineString(zip(*transformer.transform(_x, _y)))
        elif isinstance(geo, shapely.geometry.polygon.Polygon):
            _x, _y = geo.exterior.coords.xy
            return shapely.Polygon(zip(*transformer.transform(_x, _y)))

    def bezier(ls: shapely.geometry.linestring.LineString,
               interval: tuple[float, int] = 100):
        '''
        make bezier curve from shapely linestring
        '''
        def get_segs(ls):
            num_segs = len(ls.coords)-1
            segs = [
                LineString([ls.coords[i],
                            ls.coords[i+1]])
                for i in range(num_segs)
                ]
            return num_segs, segs

        def get_points(segs, step):
            return [seg.interpolate(step, normalized=True) for seg in segs]

        points = []
        _num_segs, _segs = get_segs(ls)

        if len(_segs) == 1:
            return ls

        for _step in [i/interval for i in range(interval+1)]:
            _tmp_segs = _segs
            for _ in range(_num_segs-1):
                _points = get_points(_tmp_segs, _step)
                _ls = LineString(_points)
                _, _tmp_segs = get_segs(_ls)
            points.append(_ls.interpolate(_step, normalized=True))
        return LineString(points)


class DEM():
    @staticmethod
    def get_copernicus_dem(geom: shapely.geometry.polygon.Polygon,
                           res: int = 30,
                           rprj: bool = True,
                           prj=None,
                           interp: bool = True
                           ):
        '''
        get Copernicus Global DEM from planetary computer stac catalog
        inputs:
            geom - shapely geometry (polygon / box)
            res - int: resolution of DEM (either 30, or 90, default 30)
            rprj - bool: whether or not to reprojct the dem
            prj - projection to reprject to
            interp - whether or not to interpolate nans
        returns: xarray instance of DEM
        clipped to envelope of `geom`
        if reprojected then not only clipped but also aligned
        with nan's interpolated
        '''

        validate_type(DEM.get_copernicus_dem, locals=locals())

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(collections=[f'cop-dem-glo-{res}'],
                                intersects=geom.envelope)

        items = search.item_collection()
        if len(items) > 0:
            dem = (stackstac.stack(
                planetary_computer.sign(items))
                .mean(dim='time', skipna=True)
                .squeeze()
                .rio.write_crs(4326)
                .rio.clip_box(*geom.bounds)
                )

            # for reprojecting
            # if user specifies prj (as int epsg code)
            # reproejct to that, otherwise get local utm crs
            if rprj:
                if prj:
                    if isinstance(prj, int):
                        prj = ccrs.epsg(prj)
                else:
                    prj = Utils.get_local_utm(geom)

                demprj = dem.rio.reproject(
                    prj,
                    nodata=np.nan,
                    resampling=Resampling.bilinear
                    ).rio.write_crs(prj)

                # fix wonky edges
                # trimming a few rows off of t/b l/r edge as needed
                tb1 = np.nonzero(~np.isnan(demprj.data[:, 0]))[0][0]
                tb2 = np.nonzero(~np.isnan(demprj.data[:, -1]))[0][0]
                rl1 = np.nonzero(~np.isnan(demprj.data[0, :]))[0][0]
                rl2 = np.nonzero(~np.isnan(demprj.data[-1, :]))[0][0]

                if tb1 != tb2:
                    rows = slice(*sorted([tb1, tb2]))
                else:
                    rows = slice(0, demprj.data.shape[0])

                if rl1 != rl2:
                    cols = slice(*sorted([rl1, rl2]))
                else:
                    cols = slice(0, demprj.data.shape[1])

                demprj = demprj[rows, cols]

                # interpolate NaNs
                if interp:
                    demprj.data = Utils.twoD_interp(demprj.data)

                return demprj.drop(['proj:epsg',
                                    'band',
                                    'gsd',
                                    'epsg',
                                    'platform',
                                    'proj:shape'])
            else:
                return dem
        print('could not find a DEM')
        return None


class Ridges():
    '''
    class for making ridge plots from DEMs
    inputs : shapely polygon with coords in lat/lon (epsg 4326)
    '''
    def __init__(self, aoi: shapely.Polygon, **kwargs):
        assert isinstance(
            aoi, shapely.geometry.polygon.Polygon), 'aoi not a polygon'
        self.aoi = aoi
        self.res = kwargs.get('res', 90)
        self.rprj = kwargs.get('rprj', True)

        # get local utm projection
        _gds = gpd.GeoSeries(aoi, crs=4326)
        self.prj = _gds.estimate_utm_crs()

        # unpack kwargs
        self.smooth_dict = kwargs.get('smooth_dict', {'x': 5})
        self.step = kwargs.get('step', None)
        self.cmap = kwargs.get('cmap', None)
        if self.cmap:
            if isinstance(self.cmap,
                          (ListedColormap,
                           LinearSegmentedColormap)
                          ):
                pass
                # self.cmap = self.cmap
            elif isinstance(self.cmap, str):
                self.cmap = plt.colormaps[self.cmap]
            else:
                print('reverting to default colormap')
                self.cmap = plt.colormaps['Grays']
        else:
            self.cmap = plt.colormaps['YlGnBu_r']
        self.figsize = kwargs.get('figsize', [10, 10])
        self.title = kwargs.get('title', None)
        self.vert_exag = kwargs.get('vert_exag', 0.15)
        self.fc = kwargs.get('fc', 'w')
        self.textc = kwargs.get('textc', 'w')
        self.font = kwargs.get('font', 'dejavu sans mono')

        self.get_dem()
        self.smooth()
        self.transects()
        self.plotter()

    def get_dem(self):
        '''
        get digital elevation model
        '''
        self.dem = DEM.get_copernicus_dem(geom=self.aoi,
                                          res=self.res,
                                          rprj=self.rprj,
                                          prj=self.prj)

    def smooth(self):
        # smooth DEM along x-axis
        self.dem_smooth = self.dem.rolling(self.smooth_dict,
                                           min_periods=3,
                                           center=True).mean(skipna=True)
        self.Z = self.dem_smooth.data

        # Z[np.isnan(Z)] = 0
        self.X, self.Y = np.meshgrid(self.dem_smooth.x.data,
                                     self.dem_smooth.y.data)

    def transects(self):
        '''
        sample elevation along transects
        '''
        def divide_at_nan(q):
            # for splitting transects at nan values
            groups = []
            uniquekeys = []
            for k, g in groupby(q, lambda x: np.any(np.isnan(x))):
                groups.append(list(g))    # Store group iterator as a list
                uniquekeys.append(k)
            return [groups[i] for i, tf in
                    enumerate(uniquekeys) if
                    (not tf) & (len(groups[i]) > 1)]

        # if step kwarg not supplied calcuate transect spacing
        # such that there will be 50 transects in total
        if not self.step:
            self.step = len(self.dem_smooth.y.data) // 50

        # sample transects
        # slicing x, y and z arrays
        self.z_sub = self.Z[
            [i for i in range(0, len(self.dem_smooth.y.data), self.step)],
            :
                ]
        self.x_sub = self.X[
            [i for i in range(0, len(self.dem_smooth.y.data), self.step)],
            :
                ]
        self.y_sub = self.dem_smooth.y.data[::self.step]

        # get pairs of x,z coords along each transect
        # and ensure they are sorted along x-axis
        verts = [list(zip(self.x_sub[i, :],
                          self.z_sub[i, :]))
                 for i in range(self.z_sub.shape[0])]
        verts = [sorted(v, key=lambda tup: tup[0]) for v in verts]

        # if any NaN values are encountered split the transect there
        # and resume transect at next non-NaN value
        # list of lists - some which may be lists themselves
        divided = [divide_at_nan(x) for x in verts]
        # unpack to be just a list of lists
        self.cleaned = list(chain(*divided))

        # dividing transects, increases the number of transects,
        # so need to add additional y-coordinate

        # values to plot along - i.e. duplicate y-values of transects
        # that have been split
        # if multiple splits duplicate y coords multiple times
        insert_idxs = [
            [i]*(len(c)-1) for i, c in enumerate(divided) if len(c) > 1
            ]
        insert_idxs = [val for sublist in insert_idxs for val in sublist]
        self.y_sub = np.insert(
            self.y_sub, insert_idxs, self.y_sub[insert_idxs]
            )

        # set colors
        self.facecolors = self.cmap(
            np.linspace(0, 1, len(self.y_sub))
            )

        self.edgecolors = plt.colormaps['Greys'](
            np.linspace(0, 1, len(self.y_sub))
            )

    def plotter(self):

        def polygon_under_graph(x, y, fill=0.):
            """
            Construct the vertex list which defines the polygon filling the
            space under the (x, y) line graph.
            This assumes x is in ascending order.
            """
            return [(x[0], fill), *zip(x, y), (x[-1], fill)]

        ffval = -10  # z coordinate of bottom polygon
        lw = 0.5  # line width
        alpha_p = 1

        # this sets the 'front' polygon to have lower edge of zero,
        # whereas other polygons have lower edge of ffval
        ff = np.zeros(self.y_sub.shape[0])
        ff[0:-1] = ffval

        # make polygons
        polys = [polygon_under_graph(*zip(*c),
                                     fill=ff[i])
                 for i, c in enumerate(self.cleaned)]
        self.collection = PolyCollection(polys,
                                         facecolors=self.facecolors,
                                         alpha=alpha_p,
                                         edgecolors='k',
                                         linewidths=lw)

        self.fig, self.ax = plt.subplots(figsize=self.figsize,
                                         subplot_kw={'projection': '3d'})

        self.ax.add_collection3d(self.collection, zs=self.y_sub, zdir='y')

        self.ax.set_xlim(np.nanmin(self.x_sub), np.nanmax(self.x_sub))
        self.ax.set_zlim(np.nanmin(self.z_sub), np.nanmax(self.z_sub))
        self.ax.set_ylim(np.nanmin(self.y_sub), np.nanmax(self.y_sub))

        # self.ax.set_axis_off()
        self.ax.patch.set_facecolor(self.fc)
        self.ax.yaxis.set_pane_color(self.fc)
        self.ax.xaxis.set_pane_color(self.fc)
        self.ax.zaxis.set_pane_color(self.fc)
        self.fig.patch.set_facecolor(self.fc)
        self.ax.grid(False)
        self.ax.set_box_aspect([1, 1, self.vert_exag])
        self.ax.view_init(elev=40, azim=270, roll=0)

        self.ax.set_title(self.title,
                          font=self.font,
                          fontsize=40,  # 'xx-large',
                          color=self.textc,
                          loc='left',
                          x=0.2,
                          y=0.77)

        self.ax.text(np.nanmin(self.x_sub),
                     np.nanmin(self.y_sub),
                     0.05 * np.nanmax(self.z_sub[-1, :]),
                     '  by:tlohde', 'x',
                     ha='left',
                     va='bottom',
                     fontsize=5,  # 'xx-small',
                     font='dejavu sans mono')

        self.ax.text(np.nanmax(self.x_sub),
                     np.nanmin(self.y_sub),
                     0.05 * np.nanmax(self.z_sub[-1, :]),
                     'Copernicus Global Digital Elevation Model, ESA (2021)  ',
                     'x',
                     ha='right',
                     va='bottom',
                     fontsize=5,  # 'xx-small',
                     font='dejavu sans mono')


class Flow():
    '''
    class for making flowy topography maps
    input dem
    **kwargs:
        'step' - [in crs units] size of step to take between
        samples
        reps - how many steps to take
        cmap - colormap
        N - how many points to randomly seed
    '''
    def __init__(self,
                 dem,
                 **kwargs):
        self.dem = dem
        self.aspect = xrs.aspect(dem)
        self.slope = xrs.slope(dem)

        self.epsg = self.dem.rio.crs.to_epsg()
        self.prj = ccrs.epsg(self.epsg)

        self.N = kwargs.get('N', None)
        self.step = kwargs.get('step', int(1.5 * dem.rio.resolution()[0]))
        self.reps = kwargs.get('reps', 50)
        self.gradient_threshold = kwargs.get('gradient_threshold', 5)
        self.cmap = plt.get_cmap(kwargs.get('cmap', 'twilight'))

        self.make_points()
        self.follow_aspect(step=self.step, reps=self.reps)
        self.make_line_collection()

    def make_points(self):
        '''
        randomly scatter N points across domain
        '''
        _minx, _miny, _maxx, _maxy = self.dem.rio.bounds()
        if not self.N:
            self.N = int(np.multiply(*self.dem.shape) / 10)
        self.x = np.random.uniform(_minx, _maxx, self.N)
        self.y = np.random.uniform(_miny, _maxy, self.N)

    def follow_aspect(self, step, reps):
        '''
        trace downhill path - in aspect direction
        '''
        _xpoints = self.x.copy()
        _ypoints = self.y.copy()

        _x = self.x.copy()
        _y = self.y.copy()
        for q in tqdm(range(reps)):
            x_da = xr.DataArray(_x, dims=['index'])
            y_da = xr.DataArray(_y, dims=['index'])

            theta = self.aspect.sel(
                x=x_da,
                y=y_da,
                method='nearest'
                )

            grad = self.slope.sel(
                x=x_da,
                y=y_da,
                method='nearest'
                )

            _dx = xr.where((theta.isnull())
                           | (theta <= 0)
                           | (grad.isnull())
                           | (grad < self.gradient_threshold),
                           np.nan,
                           step * np.sin(np.deg2rad(theta)))

            _dy = xr.where((theta.isnull())
                           | (theta <= 0)
                           | (grad.isnull())
                           | (grad < self.gradient_threshold),
                           np.nan,
                           step * np.cos(np.deg2rad(theta)))

            if _dy.isnull().sum().item() == _dy.shape[0]:
                # print(f'jumping out after {q}/{reps}')
                break

            _x += _dx
            _y += _dy

            _xpoints = np.vstack([_xpoints, _x])
            _ypoints = np.vstack([_ypoints, _y])

        # smooth the linestring
        self.smooth_linestrings = [
            taubin_smooth(
                LineString(
                    zip(
                        _xpoints[:, i], _ypoints[:, i]
                    )
                )
            ) for i in range(_xpoints.shape[1])
        ]

        for i, ls in enumerate(self.smooth_linestrings):
            if ls.is_valid:
                continue
            else:
                x, y = ls.coords.xy
                z = np.nonzero(np.isnan(x))[0]
                if (len(z) > 1) & (z[0] > 1):
                    self.smooth_linestrings[i] = LineString(
                        zip(
                            x[0:z[0]], y[0:z[0]]
                            )
                        )

        self.smooth_linestrings = [
            ls for ls in self.smooth_linestrings
            if ls.is_valid
            ]

        self.smooth_linestrings = [
            ls for ls in self.smooth_linestrings
            if ls.length > self.step * 3
            ]

    def make_line_collection(self):
        def LineString_to_LineCollection(ls, lsb=60):
            _sgmnts = []
            _azis = []
            _colors = []
            _alphas = []

            _cnorm = Normalize(0, 360)
            length = len(ls.coords)

            for i, ps in enumerate(
                zip(
                    ls.coords[:-1], ls.coords[1:]
                    )):

                _p1, _p2 = ps
                _sgmnts.append(
                    ([_p1, _p2])
                    )

                _xy_diff = np.array([_p2[0] - _p1[0],
                                    _p2[1] - _p1[1]])

                _azi = (90
                        - np.rad2deg(np.arctan2(_xy_diff[1],
                                                _xy_diff[0]))
                        - lsb)

                if _azi < 0:
                    _azi += 360

                _azis.append(_azi)
                _colors.append(self.cmap(_cnorm(_azi)))
                _alphas.append(1 - i/length)

            return _sgmnts, _azis, _colors, _alphas

        segments = []
        azimuths = []
        colors = []
        alphas = []

        for _ls in self.smooth_linestrings:
            seg, azi, clr, alp = LineString_to_LineCollection(_ls)
            segments += seg
            azimuths += azi
            colors += clr
            alphas += alp

        self.segments = segments
        self.azimuths = azimuths
        self.colors = colors
        self.alphas = alphas

    def plot(self, ax, lw):
        _lc = LineCollection(self.segments,
                             linewidths=lw,
                             colors=self.colors,
                             alpha=self.alphas)

        ax.add_collection(_lc)

        ax.set(xlim=(self.dem.x.min(), self.dem.x.max()),
               ylim=(self.dem.y.min(), self.dem.y.max()))


class Tanaka():
    '''
    class for constructing tanaka contours
    inputs:
    dem - digital elevation model - xarray dataarray
    method - for classifying elevation bands
        eqi - equal interval
        nbk - natural breaks
        default (eqi)
    k - number breaks (int), or list of break points
    lsb - light source bearing for illuminting the contours
    (default 300)

    '''
    def __init__(self,
                 dem: xr.core.dataarray.DataArray,
                 method: Literal["eqi", "nbk"] = "eqi",
                 k: tuple[int, list] = 10,
                 lsb: int = 300):

        self.dem = dem
        self.method = method
        self.k = k
        self.lsb = lsb
        self.contours = {'cg': None,
                         'lines': [],
                         'segments': [],
                         'azimuths': [],
                         'widths': [],
                         'alphas': []}
        self.break_reclassify()
        self.generate_contours()
        self.style_lines()

        self.epsg = self.dem.rio.crs.to_epsg()
        self.prj = ccrs.epsg(self.epsg)

    def break_reclassify(self):
        '''
        reclassifying DEM, either using user
        input (where k is a list), or
        by using xrspatial's equal_interval or natural_breaks
        '''
        if isinstance(self.k, list):
            _new_vals = list(range(len(self.k)))
            self.classif = reclassify(self.dem,
                                      bins=self.k,
                                      new_values=_new_vals)
        else:
            if self.method not in ['eqi', 'nbk']:
                raise ValueError(
                    f'{self.method} is not valid method. input \
                        either "eqi" for equal interval, \
                            or "nbk" for natural breaks')
            if self.method == 'eqi':
                self.classif = equal_interval(self.dem, k=self.k)
            elif self.method == 'nbk':
                self.classif = natural_breaks(self.dem, k=self.k)

        # construct dictionary that maps between classified groups
        # and the elevation range they span
        self.break_dict = {}
        for n in np.unique(self.classif.data):
            _min = xr.where(self.classif == n, self.dem, np.nan).min().item()
            _max = xr.where(self.classif == n, self.dem, np.nan).max().item()
            self.break_dict[n] = (_min, _max)

    def generate_contours(self):
        '''
        generates contour lines at each break
        '''
        self.contours['cg'] = contour_generator(x=self.dem.x.values,
                                                y=self.dem.y.values,
                                                z=self.dem.data)
        _breaks = [self.break_dict[0][0]] \
            + [b[1] for b in self.break_dict.values()]

        for _b in _breaks:
            self.contours['lines'] += [
                LineString(_l) for _l in self.contours['cg'].lines(_b)
                ]

    def style_lines(self):
        '''
        # for each individual contour line, split contour at
        # every node and determine azimuth
        # and from azimuth define line width
        # from contourpy docs:
            # Contour line segments are directed with higher z on the left,
            # hence closed line loops are oriented anticlockwise if
            # they enclose a region that is higher then the contour level,
            # or clockwise if they enclose a region that is lower
            # than the contour level.
            # This assumes a right-hand coordinate system.
        # i *think* this means my azimuth calcs are okay
        '''
        self.lsb = 360-self.lsb  # lsb == light source bearing
        for _line in tqdm(self.contours['lines']):

            # calculate azimuth
            for _p1, _p2 in zip(_line.coords, _line.coords[1:]):
                self.contours['segments'].append(
                    LineString([_p1, _p2])
                    )
                _xy_diff = np.array([_p2[0] - _p1[0],
                                     _p2[1] - _p1[1]])

                _azi = (90
                        - np.rad2deg(np.arctan2(_xy_diff[1],
                                                _xy_diff[0]))
                        - self.lsb)
                if _azi < 0:
                    _azi += 360

                self.contours['azimuths'].append(_azi)
                self.contours['widths'].append(
                    0.5*(0.05 + np.abs(np.cos(np.deg2rad(_azi))))
                    )

            self.contours['alphas'] += [(i+2)/len(_line.coords)
                                        for i in range(len(_line.coords)-1)]

    def plot_tanaka(self,
                    ax: matplotlib.axes._axes.Axes,
                    cmap=False):
        '''
        create line collection of tanaka contours
        and add to matplotlib axes, ax
        '''
        _segs = [list(map(tuple, zip(*s.coords.xy)))
                 for s in self.contours['segments']]

        if not cmap:
            cmap = LinearSegmentedColormap.from_list('wkw', ['w', 'k', 'w'])
        else:
            cmap = plt.get_cmap(cmap)
        _cnorm = Normalize(0, 360)
        _colors = cmap(_cnorm(self.contours['azimuths']))

        # alphas = (widths - min(widths)) / (max(widths) - min(widths))
        # if not alphas:
        #     colors[:, -1] = alphas

        _linecol = LineCollection(_segs,
                                  linewidths=self.contours['widths'],
                                  colors=_colors)

        ax.add_collection(_linecol)

        _segbounds = np.array(
            [seg.bounds for seg in self.contours['segments']]
            )
        _minx, _miny = np.min(_segbounds[:, [0, 1]], axis=0)
        _maxx, _maxy = np.max(_segbounds[:, [2, 3]], axis=0)
        ax.set_xlim(_minx, _maxx)
        ax.set_ylim(_miny, _maxy)


class SatelliteImage():
    '''
    quick grab of stack of satellite images
    holds lazy stack of satellite images
    inputs:
        `aoi` - (shapely polygon)
        `collection` - list of planetary computer stac catalogs to search
                       default: `['sentinel-2-l2a', 'landsat-c2-l2']`
        `datetime` (str) format must be "yyyy-mm-dd/yyyy-mm-dd"
        `resolution` (int) cell size in metres of returned stack
        `cloud` (int) (between 0 and 100) % cloud cover filter
        `epsg` (int) target projection
        `months` (list) - list of ints of months to include

    '''
    def __init__(self,
                 aoi: shapely.geometry.polygon.Polygon,
                 collection: list = ['sentinel-2-l2a',
                                     'landsat-c2-l2'],
                 datetime: str = "2020-01-01/2030-12-31",
                 resolution: int = 10,
                 cloud: int = 25,
                 epsg: int = None,
                 months: list = [],
                 ):

        validate_type(SatelliteImage, locals=locals())

        if not re.match(
            '^[0-9]{4}-[0-9]{2}-[0-9]{2}/[0-9]{4}-[0-9]{2}-[0-9]{2}$',
            datetime
        ):
            raise ValueError("datetime format must be 'yyyy-mm-dd/yyyy-mm/dd'")

        self.aoi = aoi
        self.collection = collection
        self.datetime = datetime
        self.resolution = resolution
        self.cloud = cloud
        self.epsg = epsg
        if self.epsg:
            self.prj = ccrs.epsg(self.epsg)
        self.months = months

        self.get_full_stack()
        self.get_stack_contains_aoi()
        self.get_least_cloudy()
        self.clip()

    def get_full_stack(self):
        '''
        get lazy instance of all items found
        in the either the chosen epsg, if supplied,
        or the most common crs in the stack
        '''
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)

        search = catalog.search(
            collections=self.collection,
            intersects=self.aoi,
            datetime=self.datetime,
            query={"eo:cloud_cover": {'lt': self.cloud}}
            )

        self.items = search.item_collection()

        # filter by months
        if len(self.months) == 0:
            print(f'found {len(self.items)} items')
        else:
            # filter items by months
            self.items = [
                item for item in self.items
                if int(item.properties['datetime'].split('-')[1])
                in self.months
                ]
            print(f'found {len(self.items)} items in months: {self.months}')

        # if target epsg not specified select most common projection in items
        # found and use that
        if not self.epsg:
            _epsgs = [item.properties['proj:epsg'] for item in self.items]
            _unq, _cnts = np.unique(_epsgs, return_counts=True)
            self.epsg = int(_unq[np.argmax(_cnts)])
            self.prj = ccrs.epsg(self.epsg)

        # create stack
        self.full_stack = stackstac.stack(
            self.items,
            epsg=self.epsg,
            resolution=self.resolution
        )

        # count of images by satellite platform (landsat / sentinel)
        self.platform_count_dict = dict(
            zip(*np.unique(self.full_stack.platform,
                           return_counts=True))
            )

    def get_stack_contains_aoi(self):
        '''
        identify items that completely contain the aoi
        (if any) and get lazy stack of those items
        '''
        _bounding_boxes = [shapely.geometry.shape(item.geometry)
                           for item in self.items]

        self.items_containing_aoi = [item for bbox, item
                                     in zip(_bounding_boxes, self.items) if
                                     bbox.contains_properly(self.aoi)
                                     ]

        if len(self.items_containing_aoi) == 0:
            print('no scenes completely contain aoi')
            self.containing_stack = None
        else:
            print(f'{len(self.items_containing_aoi)} contain aoi')
            self.containing_stack = stackstac.stack(
                self.items_containing_aoi,
                epsg=self.epsg,
                resolution=self.resolution
                )

    def get_least_cloudy(self):
        '''
        identify least cloudy item and get lazy array
        if there are scenes that completely contain aoi
        pick one of those, otherwise go for least cloudy
        of all items
        '''

        _dict = {'full': self.items,
                 'contains': self.items_containing_aoi}

        if len(self.items_containing_aoi) == 0:
            _which = 'full'
        else:
            _which = 'contains'

        self.item_least_cloud = min(
            _dict[_which],
            key=lambda item: eo.ext(item).cloud_cover
            )

        self.least_cloudy = stackstac.stack(
            self.item_least_cloud,
            epsg=self.epsg,
            resolution=self.resolution
            )

    def clip(self):
        self.clipd_full_stack = (self.full_stack
                                 .rio.clip_box(
                                    *self.aoi.bounds,
                                    crs=4326
                                    ))

        self.clipd_containing_stack = (self.containing_stack
                                       .rio.clip_box(
                                        *self.aoi.bounds,
                                        crs=4326
                                        ))

        self.clipd_least_cloudy = (self.least_cloudy
                                   .rio.clip_box(
                                    *self.aoi.bounds,
                                    crs=4326
                                    ))

    def get_bands(self,
                  which: Literal['full',
                                 'contains',
                                 'least_cloudy'] = 'contains',
                  clipped: bool = True,
                  bands: list = ['red', 'green', 'blue']):
        if clipped:
            _dict = {'full': self.clipd_full_stack,
                     'contains': self.clipd_containing_stack,
                     'least_cloudy': self.clipd_least_cloudy}
        else:
            _dict = {'full': self.full_stack,
                     'contains': self.containing_stack,
                     'least_cloudy': self.least_cloudy}

        return _dict[which].sel(band=bands)

    def get_preview(self):
        '''
        get rendered preview
        returns xarray
        '''
        preview_dict = {
            'landsat-c2-l2': 'rendered_preview',
            'sentinel-2-l2a': 'visual'
            }

        asset = preview_dict[self.item_least_cloud.collection_id]

        return rio.open_rasterio(
            (self.item_least_cloud.assets[asset].href)
            ).rio.clip_box(*self.aoi.bounds,
                           crs=4326)

    # def get_assets(self,
    #                which: Literal['full',
    #                               'contains',
    #                               'least_cloudy'] = 'contains',
    #                assets: list = ['red', 'green', 'blue'],
    #                epsg: int = None,
    #                resolution: int = None):
    #     _dict = {
    #         'full': self.items,
    #         'contains': self.items_containing_aoi,
    #         'least_cloudy': self.item_least_cloud
    #     }

    #     # check assets is valid
    #     valid_check = [asset in _dict[which].assets.keys()
    #                    for asset in assets]
    #     if np.all(valid_check):
    #         pass
    #     else:
    #         _unavailable = [asset for asset in assets if
    #                         asset not in _dict[which].assets.keys()]
    #         # print(f'assets: {_unavailable} are not available')
    #         raise KeyError(f'assets: {_unavailable} are not available')

    #     if not epsg:
    #         epsg = self.epsg
    #     if not resolution:
    #         resolution = self.resolution

    #     return stackstac.stack(_dict[which],
    #                            assets=assets,
    #                            epsg=epsg,
    #                            resolution=resolution)


class LocalCmap():
    '''
    create 'local' colormap
    from DEM and satellite imagery

    inputs:
        aoi - shapely polygon of aoi (in epsg 4326)
        levels - (int) number of elevation/colour bands
        months - (list) list of month numbers (Jan=1, Dec=12).
                 only satellite images captured in specified months
                 will be used to average colours

    '''
    def __init__(self,
                 aoi: shapely.geometry.polygon.Polygon,
                 levels: int = 12,
                 months: list = []):

        validate_type(LocalCmap, locals=locals())

        # get satellite imagery
        self.aoi = aoi
        self.SatelliteImage = SatelliteImage(self.aoi,
                                             collection=['sentinel-2-l2a'],
                                             cloud=10,
                                             months=months)

        # get dem
        self.dem = DEM.get_copernicus_dem(self.aoi,
                                          res=30,
                                          rprj=True,
                                          prj=self.SatelliteImage.prj
                                          )

        # make tanaka contours
        self.tanaka = Tanaka(self.dem, 'nbk', levels)

        self.img = self.SatelliteImage.get_preview()
        self.get_average_per_region()

    def get_average_per_region(self, func=None, **kwargs):
        '''
        average colours per elevation region
        func - function to apply to averaged output
               good ones to use:  `exposure.adjust_log`
        '''

        # reproject elevation regions to same grid as imagery
        _regions = self.tanaka.classif
        _regions = _regions.rio.reproject_match(self.img).data
        _regions = np.where(np.isnan(_regions), -1, _regions)

        _shape = self.img.shape
        _img_data = np.moveaxis(self.img.data.copy(),
                                np.argmin(_shape),
                                -1)

        # convert from rgb for averaging
        _labspace = color.rgb2lab(_img_data)

        for r in np.unique(_regions):
            idx = _regions == r
            for _band in range(_labspace.shape[-1]):
                _labspace[:, :, _band][idx] = np.median(
                    _labspace[:, :, _band][idx]
                    )

        # _avg_img = exposure.adjust_log(color.lab2rgb(_labspace), gain)
        # _avg_img = exposure.adjust_gamma(color.lab2rgb(_labspace))
        # _avg_img = exposure.equalize_hist(color.lab2rgb(_labspace))

        # convert back to rgb and apply any func
        if func:
            _avg_img = func(color.lab2rgb(_labspace), **kwargs)
        else:
            _avg_img = color.lab2rgb(_labspace)

        self.region_color_dict = {}
        for r in np.unique(_regions):
            if r not in self.tanaka.break_dict.keys():
                continue
            else:
                idx = _regions == r
                c = []
                for _band in range(_avg_img.shape[-1]):
                    c.append(np.unique(_avg_img[:, :, _band][idx]).item())
                self.region_color_dict[self.tanaka.break_dict[r][0]] = c

        self.average_colours = xr.DataArray(data=np.moveaxis(_avg_img, -1, 0),
                                            dims=self.img.dims,
                                            coords=self.img.coords)

    def make_colormap(self, ax, shrink=0.6):
        avg_boundary = list(self.region_color_dict.keys())
        avg_colors = list(self.region_color_dict.values())
        avg_cmap = ListedColormap(avg_colors)
        avg_norm = BoundaryNorm(avg_boundary, avg_cmap.N)
        avg_sm = ScalarMappable(avg_norm, avg_cmap)
        self.cax = plt.colorbar(avg_sm,
                                spacing='proportional',
                                extend='both',
                                shrink=shrink,
                                ax=ax)
        self.cax.set_ticks(avg_boundary)

    def plot(self, ax):
        self.average_colours.plot.imshow(ax=ax, rgb='band')
        self.tanaka.plot_tanaka(ax=ax)
        self.make_colormap(ax=ax)
