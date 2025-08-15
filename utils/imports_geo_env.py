import numpy as np
import matplotlib.pyplot as plt     # conda install -c conda-forge matplotlib==3.2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString, Point
import geopandas as gpd
import xarray as xr
from matplotlib.dates import DateFormatter, date2num
from glob import glob    
import pandas as pd   
from matplotlib.colors import LogNorm, BoundaryNorm, LinearSegmentedColormap
from cmcrameri import cm
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import rasterio
from rasterio.windows import from_bounds
from seaborn import set_context
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from matplotlib.patches import Wedge
from windrose import WindroseAxes
import seaborn as sns


