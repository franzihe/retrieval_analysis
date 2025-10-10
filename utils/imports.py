import xarray as xr          # conda install -c conda-forge xarray
# from pymeteo import radar    # conda install -c cwebster2 pymeteo
from glob import glob       
import numpy as np

import datetime
import IMProToo   # pip install IMProToo
import gzip    # conda install -c conda-forge gzip 
import datetime
from cmcrameri import cm            #  conda install -c conda-forge cmcrameri 
import matplotlib.pyplot as plt     # conda install -c conda-forge matplotlib==3.2
from matplotlib.dates import DateFormatter, date2num
import pandas as pd
import dask.array as da
# import functions as fct
import bz2
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup   # conda install -c anaconda beautifulsoup4 
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
import requests
import time
import seaborn as sns
from seaborn import set_context
from windrose import WindroseAxes
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.stats import linregress
from math import pi
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

