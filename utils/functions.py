# from imports import *
# from imports import requests, BeautifulSoup
# import imports
from imports import (xr, np,datetime, datetime,cm,plt,DateFormatter, pd,da, BeautifulSoup,BoundaryNorm, LinearSegmentedColormap,requests,date2num,sns, least_squares, r2_score, MinMaxScaler, Normalize, ScalarFormatter, Rectangle, Patch)
fig_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)',
             '(aa)', '(bb)', '(cc)', '(dd)', '(ee)', '(ff)', '(gg)', '(hh)', '(ii)', '(jj)', '(kk)', '(ll)', '(mm)', '(nn)', '(oo)', '(pp)', '(qq)', '(rr)', '(ss)', '(tt)', '(uu)', '(vv)', '(ww)', '(xx)', '(yy)', '(zz)']

height_bins = np.arange(0, 3000, 100) # Height bins [m] bins of 100 m for height 
ze_bins = np.arange(-10,31,1)# Reflectivity bins [dBZ] 1 dBZ for reflectivity
W_bins = np.arange(-4,4,0.075) # bin widths of 0.075 m s−1 for Doppler velocity
sW_bins = np.arange(0,1.2, 0.05) # bin widths of 0.05 m s−1 for Doppler spectral width

# cmap = cm.glasgow  # define the colormap
cmap = cm.managua
# extract all colors from the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
# cmap = LinearSegmentedColormap.from_list(
# 'Custom cmap', cmaplist, 4) # cmap.N
cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)

# custom_colors = {  
#                      'aggregate': cmap(0),  
#                      'graupel':cmap(1),   
#                      'mixed': cmap(2),
#                      'pristine':cmap(3),
                
                
#                 }
# custom_colors = {  
#                      'aggregate': cmap(1),  
#                      'graupel':cmap(4),   
#                      'mixed': cmap(2),
#                      'pristine':cmap(5),
                
                
#                 }
custom_colors = {  
                     'aggregate': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(1),  
                     'graupel':LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(4),   
                     'mixed': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(2),
                     'campaign': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(5),
                     'pristine': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(5),
                     'North': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(1),
                     'South':LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(4),   
                     'East': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(2),
                     'West': LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(5),
}

# Function to scrape file URLs
def get_file_urls(catalog_url, day):
    base_data_url = f"https://thredds.met.no/thredds/dodsC/remotesensingceilometer/2021/03/{day}/andoya/raw/"
    # base_data_url = catalog_url[:85]

    # Request the catalog page
    response = requests.get(catalog_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all NetCDF file links
    file_urls = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".nc"):
            file_urls.append(base_data_url + href)  
    return sorted(file_urls)

def clean_time(ds):
    ds = ds.sortby('time')

    # Drop invalid or missing times
    ds = ds.sel(time=~pd.isnull(ds['time'].values))

    # Remove duplicate time entries
    _, index = np.unique(ds['time'].values.astype('datetime64[ns]'), return_index=True)
    ds = ds.isel(time=sorted(index))
    
    # Ensure time is strictly increasing (monotonic)
    time_diff = np.diff(ds['time'].values)
    if np.any(time_diff <= np.timedelta64(0, 'ns')):
        raise ValueError("Non-monotonic time values still remain after cleaning.")
    
    return ds

def adjust_time(ds):
    _tt = []
    t2 = datetime.datetime.strptime(str(ds.base_time.values)[:-3], '%Y-%m-%dT%H:%M:%S.%f')
    for i in range(len(ds.time_offset)):
        t1 = datetime.datetime.strptime(str(ds.time_offset[i].values)[:-3], '%Y-%m-%dT%H:%M:%S.%f')
        _tt.append(t1 + datetime.timedelta(hours=t2.hour, minutes=t2.minute, seconds=t2.second))

    return(_tt)

def dataarray_24hours(date,array):
    _sometime = []
    for hours in np.arange(0, 24):
        _sometime.append((datetime.datetime.strptime("{} 00:00:00".format(date), '%Y%m%d %H:%M:%S') + datetime.timedelta(hours=int(hours))).strftime('%Y-%m-%d %H:%M:%S'))

    result = xr.DataArray(data = da.full(shape= (len(_sometime),), fill_value = np.nan), coords = {'time':_sometime}, )          
    result['time'] = pd.DatetimeIndex(result['time'].values)
    result = result.rename(array.name)
    result = result.assign_attrs(array.attrs)

    return(result)

def copy_all_attrs(source_ds, target_ds):
    # Copy attributes for data variables
    for var in target_ds.data_vars:
        if var in source_ds.data_vars:
            target_ds[var].attrs = source_ds[var].attrs.copy()
            
    # Copy attributes for coordinates
    for coord in target_ds.coords:
        if coord in source_ds.coords:
            target_ds[coord].attrs = source_ds[coord].attrs.copy()
            
    target_ds.attrs = source_ds.attrs.copy()
            
    return target_ds


def load_and_insert_backscatter_cbh(files, var_name_bsc, var_name_cbh, range_rounded, range_combined, layer_rounded, layer_combined, data_vars=None, drop_vars=None):
    """Loads NetCDF files, preprocesses them, and inserts backscatter into the unified range array."""
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True, preprocess = clean_time, data_vars=data_vars, drop_variables=drop_vars)
    # create a DataArray filled with NaNs aligned to that combined range. Insert backscatter data from both instruments into the correct range slots.
    # Create empty array filled with NaNs
    empty_arr_bsc = da.full(shape=(len(ds["time"]), len(range_combined)), fill_value=np.nan)
    arr_bsc = xr.DataArray(data=empty_arr_bsc, dims=("time", "altitude"), coords=dict(time=ds['time'], altitude=range_combined), name="bsc")
    
    empty_arr_cbh = da.full(shape=(len(ds["time"]), len(layer_combined)), fill_value=np.nan)
    arr_cbh = xr.DataArray(data=empty_arr_cbh, dims=("time", "layer"), coords=dict(time=ds['time'], layer=layer_combined), name="cbh")
    

    # Build index mapping from dataset-specific range values to unified range
    range_idx = np.searchsorted(range_combined, range_rounded)
    layer_idx = np.searchsorted(layer_combined, layer_rounded)
    # Assign into slices
    arr_bsc[:, range_idx] = ds[var_name_bsc].values
    arr_cbh[:, layer_idx] = ds[var_name_cbh].values
    
    arr_bsc.attrs = ds[var_name_bsc].attrs
    arr_cbh.attrs = ds[var_name_cbh].attrs
    
    ds_out = xr.merge([arr_bsc, arr_cbh]) 
    
    return ds_out

# Helper function to safely open datasets
def open_or_empty(file_list, **kwargs):
    return xr.open_mfdataset(file_list, **kwargs) if file_list else xr.Dataset()

# def plt_Ze_W_p_T_U(ds, day_start, xmax, site):

#         title = ['Airport Ceilometer backscatter','MRR reflectivity', 'MRR mean Doppler Velocity', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind']
#         fig, axsm = plt.subplots(6, 1, figsize=(8.27, 11.69), sharex=True, )
#         # fig, axsm = plt.subplots(2,1, figsize=[10,4.8], sharex=True)
#         axs = axsm.flatten()

#         ## plot lidar backscatter and cloudbase height from ceilometer
#         cmap = cm.lipari_r
#         # extract all colors from the colormap
#         cmaplist = [cmap(i) for i in range(cmap.N)]
#         # force the first color entry to be white
#         cmaplist[0] = (255,255,255, 0.9)
#         # create the new map
#         cmap = LinearSegmentedColormap.from_list(
#         'Custom cmap', cmaplist, 10) # cmap.N
#         # define the bins and normalize
#         bounds = np.linspace(0., 7e7, 11)
#         norm = BoundaryNorm(bounds, cmap.N)
#         cf0 = ds['bsc'].plot.pcolormesh(ax=axs[0], x='time', cmap=cmap, norm=norm, vmin=0., add_colorbar = False)
#         cbaxes = fig.add_axes([0.92, 0.82, 0.015, 0.125])
#         cbar = plt.colorbar(cf0, cax=cbaxes, shrink=0.5, extend='max', orientation='vertical')
#         cbar.ax.set_ylabel('Backscatter')
        
#         (ds['cbh'].isel(layer=0)).plot.line(ax=axs[0], x = 'time', linestyle="",marker=".",markersize=1.5, alpha=0.3, color = 'grey', add_legend=True, label='Cloud Base',)
        
#         axs[0].set_yticks((ds['height_c'].values[1::100]).round())
#         axs[0].set(ylabel = "Height [m]", yticklabels=(ds['height_c'].values[1::100]).round())
#         axs[0].set_ylim([0,5000])
#         axs[0].legend()

#         ## first plot reflectivity
#         cf1 = ds['Ze'].plot.pcolormesh(ax = axs[1], 
#                                 x='time', 
#                                 vmin=-5,vmax=20,# levels=np.arange(-5,20),# levels=np.arange(-20,30,0.5), 
#                                 cmap=cm.hawaii_r,#plt.get_cmap('pymeteo_radar'), 
#                                 extend='both', add_colorbar=False)
#         # cbaxes = fig.add_axes([0.92, 0.8, 0.025, 0.14])
#         cbaxes = fig.add_axes([0.92, 0.675, 0.015, 0.125])
#         cbar = plt.colorbar(cf1, cax=cbaxes, shrink=0.5, extend='both', orientation='vertical')
#         cbar.ax.locator_params(nbins=7)
#         cbar.ax.set_ylabel('Ze [{}]'.format(ds['Ze'].attrs['units']))

#         ## second plot Doppler velocity
#         cf2 = ds['W'].plot.pcolormesh(ax = axs[2], 
#                                         x='time',
#                                         vmin=-3, vmax=3, #levels=np.arange(-3,3.1,0.1), 
#                                         extend='both', cmap=cm.vik, add_colorbar=False,)
#         # cbaxes = fig.add_axes([0.92, 0.625, 0.025, 0.14])
#         cbaxes = fig.add_axes([0.92, 0.53, 0.015, 0.125])
#         cbar = plt.colorbar(cf2, cax=cbaxes, shrink=0.5, extend='both', orientation='vertical')
#         cbar.ax.locator_params(nbins=6)

#         cbar.ax.set_ylabel('W [{}]'.format(ds['W'].attrs['units']))

#         ## third plot precipitation accumulation hourly
#         precip = ds['precipitation'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)
#         # Plot valid precipitation as a bar chart
#         axs[3].bar(precip.time.values, precip.values, width=np.timedelta64(40, 'm'), align='center', color='blue', label='Valid Precipitation', alpha=0.5)
        
#         # Highlight missing data periods with shaded background
#         missing_times = precip['time'][np.where(precip==0.)].values
#         for t in missing_times:
#             axs[3].axvspan(t- np.timedelta64(20, 'm'), t + np.timedelta64(20, 'm'), color='gray', alpha=0.2)
#         axs[3].set_ylabel('Precipitation [{}]'.format(ds['precipitation'].attrs['units']))
#         axs[3].set_ylim([0, 4])
#         axs[3].legend()

#         ## forth plot 2-m temperature
#         if len(ds['air_temperature'].dropna(dim='time')) != 0:
#                 ds['air_temperature'].dropna(dim='time').plot(ax = axs[4], x='time')
#                 axs[4].hlines(y= 2., xmin=day_start, xmax=xmax, color = 'grey')
        
#         axs[4].set_ylabel('2-m temperature [{}]'.format(ds['air_temperature'].attrs['units']))
#         axs[4].set_ylim([-11.5, 12.5])

#         ## fifth plot 10-m wind
#         color = 'tab:blue'
#         if len(ds['wind_direction'].dropna(dim='time')) !=0:
#                 axs[5].scatter(x=ds['wind_direction'].dropna(dim='time').time.values, y =ds['wind_direction'].dropna(dim='time').values, color=color,s=5)
#         axs[5].set_ylabel('Wind direction [{}]'.format(ds['wind_direction'].attrs['units']),color=color, )
#         axs[5].tick_params(axis='y', labelcolor=color)
#         axs[5].set_ylim(0, 360)

#         ax2 = axs[5].twinx()
#         color = 'tab:green'
#         if len(ds['wind_speed'].dropna(dim='time')) !=0:
#                 ds['wind_speed'].dropna(dim='time').plot(ax=ax2, x='time', color=color)
#         ax2.set_ylabel('Wind speed [{}]'.format(ds['wind_speed'].attrs['units']), color=color)
#         ax2.tick_params(axis='y', labelcolor=color)
#         ax2.set_ylim(0,20)

#         # Define the date format
#         date_form = DateFormatter("%H:00")
#         # if len(ds['air_temperature'].dropna(dim='time')) != 0
#         for ax, tt in zip(axs,title):
#                 ax.set( xlabel = "",
#                                 title =tt)
#                 ax.xaxis.grid()
#                 ax.tick_params(axis='x', labelrotation=0, )
#                 plt.setp(ax.get_xticklabels(), horizontalalignment="center")
#                 ax.xaxis.set_major_formatter(date_form)
#                 if len(ds['air_temperature'].dropna(dim='time')) != 0 or len(ds['wind_direction'].dropna(dim='time')) !=0 or len(ds['wind_speed'].dropna(dim='time')) !=0:
#                         ax.set_xlim([day_start, xmax])
#         for ax in axs[1:3]:
#             # ax.set_yticks((10, 20, 30))
#             ax.set_yticks(ds['range'].values[1::5])
#             ax.set(ylabel = "Height [m]", yticklabels=ds['height'].isel(time=0).values[1::5])
#             # ax.set(ylabel = "Height [m]", yticklabels=ds['height'].isel(time=0).sel(range=slice(10,30,10)).values)
#         axs[-1].set(xlabel = "Time [UTC]",)
            
#         fig.suptitle("{} ({})".format(site.split('.')[0], ds.time.dt.date[0].values), fontweight="bold");
#         fig.subplots_adjust(top=0.94)


def class_to_flag_array(most_common_class, class_name):
        return xr.where(most_common_class == class_name, 1.0, np.nan).rename(class_name)

def empty_class_array(name, times):
        return xr.DataArray(
            data=np.full(len(times), np.nan),
            coords={'time': times},
            dims=['time'],
            name=name
        )
        
def arrays_same_time_res(variable, da):
    """
    Align variable to target time grid using nearest neighbour.
    Returns a NEW DataArray.
    """
    # Allocate fresh array (critical!)
    out = xr.full_like(da, np.nan)

    # Get the nearest indices
    nearest_indices = da.indexes['time'].get_indexer(variable.time.values, method='nearest')
    
    # Assign values to the appropriate time index
    out.data[nearest_indices] = variable.data
    
    return out

def fill_10min_to_1min(ds, var_name):
    # Fill the Mean wind direction over the last ten minutes before the observation time
    var_filled = ds[var_name].copy()

    for t in ds.time.values:
        val = ds[var_name].sel(time=t).values
        if not np.isnan(val):
            # print(val)
            # Assign to [t - 9min, ..., t]
            time_window = (ds.time >= t - np.timedelta64(9, "m")) & (ds.time <= t)
            var_filled.loc[time_window] = val
            
    return var_filled

# def MRR_bin_averaged_dataset(ds, dsm):
#     ## Average MRR variables in range bins
#     # Radar equivalent reflectivity ($Ze$) in dBz is transformed to MRR radar reflectivity ($Z$) 
#     # in $mm^6 m^{-3}$ by the following relationship:
#     # # $$Ze = 10log_{10} (\frac{Z}{mm^6 m^{-3}})$$
#     # # $$Z = 10^{Ze/10} mm^6 m^{-3}$$
#     # # Once we know $Z$ it is possible to average over range sizes. The assumption is that the 4th 
#     # range represents the bin between 300 and 400m and so on. Reflectivity, Doppler velocity, and height will 
#     # be averaged in the bins (4,5), (6,7), (8,9), ... (26,27), (28,29), (30,31)
    
#     _Ze_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])), fill_value=np.nan), coords ={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
#     _W_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])), fill_value=np.nan), coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
#     _height_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])),fill_value=np.nan),coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
#     _sWidth_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])),fill_value=np.nan),coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
#     for i, k in zip(ds.range[1:-1][::2].values, range(len(ds.range[1:-1][::2]))):
#                 _Z = 10**(ds['Ze'].sel(range=slice(i, (i+1)))/10)
#                 _Z_mean = _Z.mean('range', skipna=True, keep_attrs=True)
#                 _Ze_mean[:, k] = 10*np.log10(_Z_mean)

#                 _W_mean[:,k] = ds['W'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)
#                 _height_mean[:,k] = ds['height'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)
#                 _sWidth_mean[:,k] = ds['spectralWidth'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)


#     dsm['Ze'] = _Ze_mean
#     dsm['W']  = _W_mean
#     dsm['height'] = _height_mean
#     dsm['spectralWidth'] = _sWidth_mean
        
#     dsm['Ze'].attrs = ds['Ze'].attrs
#     dsm['W'].attrs = ds['W'].attrs
#     dsm['height'].attrs = ds['height'].attrs
#     dsm['spectralWidth'].attrs = ds['spectralWidth'].attrs

#     return dsm


def extract_clean_data(ds, variables, mask=None):
    result = {}
    for var in variables:
        vals = ds[var]
        if mask is not None:
            vals = vals.where(mask)
        vals = vals.values.flatten()
        vals = vals[~np.isnan(vals)]
        result[var] = vals
    return result

# Custom function to convert meteorological degrees (0° N, clockwise)
# to matplotlib degrees (0° E, counterclockwise)
def met_to_math_deg(theta_met):
    return (90 - theta_met) % 360

def probability_distribution(ds, var_name1, var_name2, var_bins1, var_bins2):
    """
    Create a 2D histogram of the specified variable and height.
    """
    is_2d = ds[var_name2].ndim == 2
    if is_2d != True:
        # print(f"Create 2D of variable {var_name2}")
        var2_1d = ds[var_name2].values
        time_len, var2_len = ds[var_name1].shape
        var2_2d = np.broadcast_to(var2_1d, (time_len, var2_len))
        var2 = var2_2d.flatten()
    else:
        var2 = ds[var_name2].values.flatten()

    # Remove Nans and flatten the data
    var1 = ds[var_name1].values.flatten()
    
    mask = ~np.isnan(var1)
    var1 = var1[mask]
    var2 = var2[mask]
    
    # 2D histogram (probability density)
    H_var, xedges, yedges = np.histogram2d(var2, var1, bins=[var_bins2, var_bins1])
    
    
    # Normalize the histogram to get a probability distribution
    H_var_prob = H_var / np.sum(H_var)
    
    return H_var, H_var_prob

def plt_prob_matrix(height_bins,
        *,
        ze   : dict,    # {'marine': H_ze_prob_marine , 'wai': … , 'other': …}
        w    : dict,    # same keys, Doppler velocity probability
        sw   : dict,    # same keys, spectral-width probability
        counts_ze : dict,  # raw‐count matrices for titles
        counts_w  : dict,
        counts_sw : dict,
        include_other=True,
        fig_size=(9, 9)):
    
    """
    Draw probability heat-maps for Ze / W / sW.
    -------------------------------------------------------
    height_bins : 1-D array of height/range bin edges
    ze, w, sw   : dicts of 2-D probability arrays (shape = [H, Z-bins])
                  keys must contain 'marine' and 'wai'; 'other' optional
    counts_*    : corresponding raw-count arrays (for subtitle totals)
    include_other : if False → only 'marine' & 'wai' columns are drawn
    """
    
    # & colour limits
    categories = ['marine', 'wai'] + (['other'] if include_other else [])
    colours    = {'marine': 'mediumblue', 'wai': 'crimson', 'other': 'darkorange'}
    titles     = {'marine': 'Marine', 'wai': 'WAI', 'other': 'Other'}
        
    # variable-specific x-bins 
    xbins_dict = {
            # 'Ze': np.arange(0., 0.003, 0.0002),
            'Ze': ze_bins,
            'W' : W_bins,          # use same for demo
            'sW': sW_bins
    }

    bounds_dict = {
        'Ze': np.arange(0., 0.003, 0.0002),
        'W' : np.arange(0., 0.003, 0.0002),          # use same for demo
        'sW': np.arange(0., 0.0085, 0.0005)
    }
        
    # data containers in order of rows
    var_data   = {'Ze': ze, 'W': w, 'sW': sw}
    var_counts = {'Ze': counts_ze, 'W': counts_w, 'sW': counts_sw}

    n_cols = len(categories)
    cmap   = cm.davos_r# define the colormap

        
    fig, axs = plt.subplots(
            nrows=3, ncols=n_cols,
            figsize=fig_size, sharey=True
        )
    axs = np.atleast_2d(axs)    # safe even if n_cols=2
        
    # iterate over rows (variables) and columns (categories)
    for row, (var, datadict) in enumerate(var_data.items()):
        # Define discrete levels
        bounds = bounds_dict[var] #bounds = np.arange(0.,0.0085, 0.0005)
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        
        for col, cat in enumerate(categories):
            ax = axs[row, col]
            pcm = ax.pcolormesh(xbins_dict[var], height_bins, datadict[cat], cmap=cmap, norm = norm)
            
            # x-axis label only on bottom row
            # if row == 2:
            #     ax.set_xlabel(f'{var} bin')
            if col == 0:
                ax.set_ylabel('Height (m)')
                
            # title with raw counts
            total = int(np.nansum(var_counts[var][cat]))
            ax.set_title(f"{titles[cat]}\n(counts={total})", fontweight='bold', color=colours[cat])
            

            
            ax.grid(True, ls=':')
            if row == 0:
                ax.set_xlabel('Reflectivity Ze (dBZ)')
            if row == 1:
                ax.set_xlabel('Doppler Velocity (m/s)')
            if row == 2: 
                ax.set_xlabel('Spectral Width (m/s)')
                
            
        # add a single color-bar per row (right-most position)
        cax = fig.add_axes([
                0.92,                # left
                0.70 - row * 0.33,   # bottom
                0.015,               # width
                0.25                 # height
            ])
        fig.colorbar(
                pcm, cax=cax, orientation='vertical',
                label='Probability'
            )
        
    plt.tight_layout(rect=[0, 0, 0.9, 1])   # leave room for cbars
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.08, 0.9, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    return fig, axs

def plt_joint_prob_matrix(*,
        ze_w   : dict,    # {'marine': 2D array, ...} for Ze vs W
        ze_sw  : dict,    # Ze vs spectral width
        w_sw   : dict,    # W vs spectral width
        counts_ze_w : dict,  # raw‐count matrices for titles
        counts_ze_sw  : dict,
        counts_w_sw : dict,
        include_other=True,
        fig_size=(9, 9)):
    
    # & colour limits
    categories = ['marine', 'wai'] + (['other'] if include_other else [])
    colours    = {'marine': 'mediumblue', 'wai': 'crimson', 'other': 'darkorange'}
    titles     = {'marine': 'Marine', 'wai': 'WAI', 'other': 'Other'}
        
    # variable-specific x-bins 
    xbins_dict = {
            # 'Ze': np.arange(0., 0.003, 0.0002),
            'ze_w': ze_bins,
            'ze_sw' : ze_bins,          # use same for demo
            'w_sw': W_bins
    }
    ybins_dict = {
            'ze_w': W_bins,
            'ze_sw': sW_bins,
            'w_sw': sW_bins
    }

    bounds_dict = {
        'ze_w': np.arange(0., 0.003, 0.0002),
        'ze_sw' : np.arange(0., 0.016, 0.001),          # use same for demo
        'w_sw': np.arange(0.,0.0085, 0.0005)
    }
        
    # data containers in order of rows
    var_data   = {'ze_w': ze_w, 'ze_sw': ze_sw, 'w_sw': w_sw}
    var_counts = {'ze_w': counts_ze_w, 'ze_sw': counts_ze_sw, 'w_sw': counts_w_sw}

    n_cols = len(categories)
    cmap   = cm.davos_r# define the colormap

        
    fig, axs = plt.subplots(
            nrows=3, ncols=n_cols,
            figsize=fig_size, sharex=False, sharey=False,
        )
    axs = np.atleast_2d(axs)    # safe even if n_cols=2
        
    # iterate over rows (variables) and columns (categories)
    for row, (var, datadict) in enumerate(var_data.items()):

        # Define discrete levels
        bounds = bounds_dict[var] #bounds = np.arange(0.,0.0085, 0.0005)
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        
        for col, cat in enumerate(categories):
            ax = axs[row, col]
            pcm = ax.pcolormesh(xbins_dict[var], ybins_dict[var], datadict[cat], cmap=cmap, norm = norm)
            
            # x-axis label only on bottom row
            # if row == 2:
            #     ax.set_xlabel(f'{var} bin')
            ax.set_xlabel({'ze_w': 'Ze (dBz)', 'ze_sw': 'Ze (dBz)', 'w_sw': 'Doppler velocity (m/s)'}[var])
            if col == 0:
                ax.set_ylabel({'ze_w': 'Doppler Velocity (m/s)', 'ze_sw': 'spectral Width (m/s)', 'w_sw': 'spectral Width (m/s)'}[var])
                
            # title with raw counts
            total = int(np.nansum(var_counts[var][cat]))
            ax.set_title(f"{titles[cat]}\n(counts={total})", fontweight='bold', color=colours[cat])
            

            
            ax.grid(True, ls=':')

        # add a single color-bar per row (right-most position)
        cax = fig.add_axes([
                0.92,                # left
                0.70 - row * 0.33,   # bottom
                0.015,               # width
                0.25                 # height
            ])
        fig.colorbar(
                pcm, cax=cax, orientation='vertical',
                label='Probability'
            )
        
    plt.tight_layout(rect=[0, 0, 0.9, 1])   # leave room for cbars
    return fig, axs

def plt_prob_matrix_with_diff(height_bins,
        *,
        ze   : dict,  # 'marine', 'wai', 'diff'
        w    : dict,
        sw   : dict,
        counts_ze : dict,  # raw‐count matrices for titles
        counts_w  : dict,
        counts_sw : dict,
        fig_size=(9, 9)):

    """
    Draw 3×3 grid of probability heatmaps for Ze, W, sW:
    Columns: Marine | WAI | Marine - WAI
    """
    
    # variable-specific x-bins 
    xbins_dict = {
                # 'Ze': np.arange(0., 0.003, 0.0002),
                'Ze': ze_bins,
                'W' : W_bins,          # use same for demo
                'sW': sW_bins
        }
    
    bounds_dict = {
        'Ze': np.arange(0., 0.003, 0.0002),
        'W' : np.arange(0., 0.003, 0.0002),          # use same for demo
        'sW': np.arange(0., 0.0095, 0.0005)
    }
    
    bounds_dict_diff = {
        'Ze': np.arange(-1.2*10**(-3), 1.3*10**(-3), 0.1*10**(-3)),
        'W' : np.arange(-1.2*10**(-3), 1.3*10**(-3), 0.1*10**(-3)),
        'sW': np.arange(-4*10**(-3), 4.1*10**(-3), 0.5*10**(-3))
    }
    
    var_data = {'Ze': ze, 'W': w, 'sW': sw}
    var_counts = {'Ze': counts_ze, 'W': counts_w, 'sW': counts_sw}
    cmap_prob = cm.davos_r
    cmap_diff = cm.vik # diverging colormap for diffs
    
    
    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=fig_size,sharey=True)
    axs = np.atleast_2d(axs)
    
    for row, var in enumerate(var_data):
        xbins = xbins_dict[var]
        data = var_data[var]
        
        
        # Regular probability colormap
        bounds = bounds_dict[var]
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        
        # Difference colormap
        bounds_diff = bounds_dict_diff[var]
        norm_diff = BoundaryNorm(boundaries=bounds_diff, ncolors=256)
        
        # ── Marine ─────────────────────
        pcm1 = axs[row, 0].pcolormesh(xbins, height_bins, data['marine'], cmap=cmap_prob, norm=norm)
        axs[row, 0].set_ylabel("Height (m)")
        total = int(np.nansum(var_counts[var]['marine']))
        axs[row, 0].set_title(f"Marine \n(counts={total})",color='mediumblue', fontweight='bold')
        
        # ── WAI ───────────────────────
        axs[row, 1].pcolormesh(xbins, height_bins, data['wai'], cmap=cmap_prob, norm=norm)
        total = int(np.nansum(var_counts[var]['wai']))
        axs[row, 1].set_title(f"WAI \n(counts={total})", color='crimson', fontweight='bold')
        
        # ── Difference ────────────────
        diff = data['diff'].copy()
        diff[np.where(diff == 0)] = np.nan
        pcm2 = axs[row, 2].pcolormesh(xbins, height_bins, diff, cmap=cmap_diff, norm=norm_diff)
        axs[row, 2].set_title("Marine - WAI", fontweight='bold')
        
        # ── Colorbars ────────────────
        cax1 = fig.add_axes([0.9, 0.745 - row * 0.33, 0.015, 0.2])
        fig.colorbar(pcm1, cax=cax1, orientation='vertical', label='Probability')

        cax2 = fig.add_axes([1.05, 0.745 - row * 0.33, 0.015, 0.2])
        fig.colorbar(pcm2, cax=cax2, orientation='vertical', label='Diff', extend='both')
        
        
        for col in range(3):
            axs[row, col].grid(True, ls=':')
            if row == 0:
                axs[row, col].set_xlabel('Reflectivity Ze (dBZ)')
            if row == 1:
                axs[row, col].set_xlabel('Doppler Velocity (m/s)')
            if row == 2: 
                axs[row, col].set_xlabel('Spectral Width (m/s)')
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.08, 0.9, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    return fig, axs

def noise_corrected_ze(da):
    """
    Return a copy of `da` (Ze) in which range‑bins 21‑25 are set to NaN
    *after* 2023‑03‑07 10:32 **if** bin‑20 is NaN in that minute.
    """
    # split in time
    zcorr1 = da.sel(time=slice(None, '2023-03-07T10:32')).copy()
    zcorr2 = da.sel(time=slice('2023-03-07T10:33', None)).copy()
    # mask range 21-25 if range 20 is Nan
    mask_time = zcorr2.isel(range=20).isnull()                 # 1‑D (time) mask
    # only keep it for the range slice 21‑25
    # Now set range bins 21–25 to NaN at those times
    zcorr2.loc[dict(time=mask_time, range=slice(21, 25))] = np.nan

    # mask is True, and for range 21-25, set values to Nan 
    # concatenate back along time (keeps coordinates & attrs)
    return xr.concat([zcorr1, zcorr2], dim='time').sortby('time')

# ### Use the empirical Ze-S relationship as given in 
# Maahn, M. and Kollias, P.: Improved Micro Rain Radar snow measurements using Doppler spectra post-processing, Atmos. Meas. Tech., 5, 2661–2673, https://doi.org/10.5194/amt-5-2661-2012, 2012. 

# $$ Ze = 56 \cdot S^{1.2}  \rightarrow S = \left(\frac{Ze}{56}\right)^{1/1.2}$$

# assume Ze in $mm^6/m^3$ to get from [dBz] to [$mm^6/m^3$]

# $$Ze [dBz] = 10 \cdot log_{10}(Ze [mm^6/m^3])$$
# $$\rightarrow Ze [mm^6/m^3] = 10 ^{dBz/10}$$
# Convert Ze (dBz) to snowfall rate (S)
def ze_to_snowfall_rate(Ze_dbz, a=56, b=1.2):
    # Ze to linear units
    Ze_lin = 10 ** (Ze_dbz /10)
    # Apply Z-S relationship
    S = (Ze_lin / 56) ** (1/1.2)                # Matrosov (2007) - aggregate snowflakes
    S_bracci = (Ze_lin / 134) ** (1/1.25)       # Bracci et al. (2021) - aggregate
    S_nakai_a = (Ze_lin / 107.4) ** (1/1.67)    # Nakai et al. (2021) - heavily rimed aggregate
    S_nakai_g = (Ze_lin / 227.4) ** (1/1.67)    # Nakai et al. (2021) - graupel
    S_pu = (Ze_lin / 252) ** (1/2.03)           # Pu et al. (2020) - graupel
    S_scho = (Ze_lin / 77.61) **(1/1.22)        # Schoger et al. 2021 - Hyttalia
    S_ensemble = (Ze_lin/a) ** (1/b)


    # convert rate per minute to mm/h
    S = (S/60)#.resample(time='1H').sum()
    S_bracci = (S_bracci/60)#.resample(time='1H').sum()
    S_nakai_a = (S_nakai_a/60)#.resample(time='1H').sum()
    S_nakai_g = (S_nakai_g/60)#.resample(time='1H').sum()
    S_pu = (S_pu/60)#.resample(time='1H').sum()
    S_scho = (S_scho/60)
    S_ensemble = (S_ensemble/60)
    
    # df1 = S.isel(range=3).drop('range').rename('Matrosov_aggregate').to_dataframe()
    df1 = S.rename('Matrosov_aggregate').to_dataframe()
    # df2 = S_bracci.isel(range=3).drop('range').rename('Bracci_aggregate').to_dataframe()
    df2 = S_bracci.rename('Bracci_aggregate').to_dataframe()
    # df3 = S_nakai_a.isel(range=3).drop('range').rename('Nakai_aggregate').to_dataframe()
    df3 = S_nakai_a.rename('Nakai_aggregate').to_dataframe()
    # df4 = S_nakai_g.isel(range=3).drop('range').rename('Nakai_graupel').to_dataframe()
    df4 = S_nakai_g.rename('Nakai_graupel').to_dataframe()
    # df5 = S_pu.isel(range=3).drop('range').rename('Pu_graupel').to_dataframe()
    df5 = S_pu.rename('Pu_graupel').to_dataframe()
    df7 = S_scho.rename('Schoger').to_dataframe()
    df6 = S_ensemble.rename(f'S_{a}_{b}').to_dataframe()
    
    combined_df = pd.concat([df1, df2, df3, df4, df5, df7, #df6
                             ], axis=1) # Use axis=1 to combine as new columns
    return  combined_df # in mm/hr


def get_subplot_title(day_start):
    
    if pd.Timestamp("2023-02-23") <= day_start <= pd.Timestamp("2023-03-13"):
        # titles = ["CL61 Ceilometer backscatter",'MRR reflectivity', 'MRR mean Doppler Velocity', 'MRR spectral width', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind', 'Snowflake class (MASCDB)']
        titles = ['MRR reflectivity', 'Snowflake class (MASCDB)', 'Precipitation accumulation', 'Reflectivity at 400m', 'Reflectivity linear at 400m', ]
    elif pd.Timestamp("2023-02-12") <= day_start <= pd.Timestamp("2023-02-22") or day_start == pd.Timestamp("2023-03-14"):
        titles = ['Airport Ceilometer backscatter','MRR reflectivity', 'MRR mean Doppler Velocity', 'MRR spectral width', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind', 'Snowflake class (MASCDB)']
    else:
        titles = ['Airport Ceilometer backscatter','MRR reflectivity', 'MRR mean Doppler Velocity', 'MRR spectral width', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind', 'Snowflake class (MASCDB)']
    
    return titles


def plot_pcolormesh_with_cbar(ax, fig, data, x, y, cmap, norm, cbar_bounds, cbar_label, cbar_pos):
    pcm = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, shading='auto')
    cbaxes = fig.add_axes(cbar_pos)
    cbar = plt.colorbar(pcm, cax=cbaxes, shrink=0.5, extend='both', orientation='vertical')
    cbar.ax.set_ylabel(cbar_label)
    if cbar_bounds is not None:
        cbar.set_ticks(cbar_bounds)
    return pcm


def format_range_axis(ax, ds):
    ax.set_yticks(ds['range'][::9])
    ax.set_yticklabels(ds['height'][0].values[::9])
    ax.set_ylim(0, ds['range'].max())
    ax.set_ylabel("Height [m]")


def plot_snowflake_classes(ax, ds):
    particle_classes = [
    ('aggregate', 1),
    # ('columnar_crystal', 2),
    # ('columnar_planar_combination', 3),
    ('graupel', 2),
    ('mixed', 3),
    ('pristine', 4)
    # ('planar_crystal', 5),
    # ('small_particle', 6)
]
    markers = ['*', '^',  'h','o',  's', 'd']
    counts = [] 

    for y_pos, (name, level) in enumerate(particle_classes, 1) :
        # data = ds[name].dropna('time')
        data = (ds['most_common_class'].where(ds['most_common_class'] == name)).dropna('time')
        counts.append(len(data))

        ax.plot(date2num(data['time']), 
                np.full(len(data['time']), level),
                marker=markers[y_pos - 1], linestyle='None',
                color=custom_colors[name], markersize=12, label=name, alpha=0.7)
    # ax.legend(loc='best', )#fontsize=8)
    ax.set_yticks(range(1, 5))
    # ax.set_yticklabels("") 
    ax.set_yticklabels([f"{name} ({cnt})" for (name, _), cnt in zip(particle_classes, counts)])
    ax.grid(True, ls=':')



def plt_Ze_W_p_T_U(ds, day_start, xmax, site):
    # Colorbar layout parameters
    cbar_left = 0.92
    cbar_width = 0.015
    # cbar_height = 0.0935
    cbar_height = 0.2
    # cbar_top = 0.85  # starting point at the top
    cbar_top = 0.7
    cbar_step = 0.1075  # vertical space between cbar axes
        
    

    # titles = ['Airport Ceilometer backscatter','MRR reflectivity', 'MRR mean Doppler Velocity', 'MRR spectral width', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind', 'Snowflake class (MASCDB)']
    titles = get_subplot_title(day_start)
    # fig, axsm = plt.subplots(5, 1, figsize=(8.27, 11.69), sharex=True, )
    fig, axsm = plt.subplots(3, 1, figsize=(8.27, 7.014/3*3), sharex=True, )
    axs = axsm.flatten()
    
    
    
    time_vals = date2num(ds['time'].values)
    # Ceilometer
    cmap = cm.lipari_r  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be white
    cmaplist[0] = (255,255,255, 0.9)
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 10) # cmap.N
    # define the bins and normalize
    # bounds = np.linspace(0, 7e7, 11)
    bounds = np.linspace(0, 200, 9)
    norm = BoundaryNorm(bounds, cmap.N)
    


    
    # Reflectivity'
    # Define your radar-style reflectivity levels (can match the image closely)
    bounds  = np.array([0, 3.5, 7, 10.5, 14, 17.5, 21, 24.5, 28])
    norm    = BoundaryNorm(boundaries=bounds, ncolors=256)  # hawaii_r has 256 levels
    # Use Crameri colormap (hawaii_r)
    cmap    = cm.hawaii_r
 

    plot_pcolormesh_with_cbar(ax=axs[0], fig=fig, data=ds['Ze'].transpose('range','time'), x=time_vals, y=ds['range'], cmap=cmap, norm=norm, cbar_bounds=bounds, cbar_label=f"Reflectivity [{ds['Ze'].attrs['units']}]", cbar_pos=[cbar_left, cbar_top -0 *cbar_step, cbar_width, cbar_height])
    axs[0].hlines(4., ds.time.values[0], ds.time.values[-1], colors='k', linestyles='-.', label='400m')

    format_range_axis(axs[0], ds)

    # snowflake class
    plot_snowflake_classes(axs[1], ds)

    
    # Precipitation accumulation hourly
    precip = ds['precipitation'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)
    axs[2].bar(precip.time.values, precip.values, width=np.timedelta64(40, 'm'), align='center', color='tab:blue', label='Geonor', alpha=0.5)
    # Highlight missing data periods with shaded background
    missing_times = precip['time'][np.where(precip==0.)].values
    for t in missing_times:
            axs[2].axvspan(t- np.timedelta64(20, 'm'), t + np.timedelta64(20, 'm'), color='gray', alpha=0.2)

    bb = (fig.subplotpars.left -0.1, fig.subplotpars.bottom-0.23, 
        fig.subplotpars.right-fig.subplotpars.left + 0.2,.1)
    precip_m = ds['Matrosov_a_400'].resample(time='1H').sum('time',skipna=True, keep_attrs=True)/1 # mm/h
    axs[2].plot(precip_m.time.values, precip_m.values, color='darkblue', label="Matrosov (2007): Z = 56 S$^{1.2}$", linestyle='solid')
    
    # precip_m = ds['Bracci_a_400'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)/1 # mm/h
    # axs[2].plot(precip_m.time.values, precip_m.values, color='darkblue', label="Bracci et al. (2022): Z = 134 S$^{1.25}$", linestyle = 'dotted')

    precip_m = ds['Nakai_a_400'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)/1 # mm/h
    axs[2].plot(precip_m.time.values, precip_m.values, color='darkblue', label="Nakai et al. (2022): Z = 107.4 S$^{1.67}$", linestyle = 'dashed')

    precip_m = ds['Nakai_g_400'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)/1 # mm/h
    axs[2].plot(precip_m.time.values, precip_m.values, color='darkblue', label = "Nakai et al. (2022): Z = 227.4 S$^{1.67}$", linestyle = 'dashdot')

    # precip_m = ds['Pu_g_400'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)/1 # mm/h
    # axs[2].plot(precip_m.time.values, precip_m.values, color='darkblue', label = "Pu et al. (2020): Z = 252 S$^{2.03}$", linestyle = (0, (3, 1, 1, 1)))

    axs[2].set_ylabel(f"Precipitation [{ds['precipitation'].attrs['units']}]")
    axs[2].set_ylim([0, 4.1])
    axs[2].legend(bbox_to_anchor=bb, mode="expand", loc="lower left",
                ncol=2, borderaxespad=0., bbox_transform=fig.transFigure)


#     # Doppler velocity
#     bounds  = np.arange(-6,7,1)
#     norm    = BoundaryNorm(boundaries=bounds, ncolors=256)
#     cmap    = cm.vik
    
# #     dopp    = axs[2].pcolormesh(time_vals, ds['range'], ds['W'].transpose('range', 'time'), cmap=cmap, norm=norm, shading='auto')
#     plot_pcolormesh_with_cbar(ax=axs[3], fig=fig, data=ds['W'].transpose('range', 'time'), x=time_vals, y=ds['range'], cmap=cmap, norm=norm, cbar_bounds=np.arange(-6, 8, 2), cbar_label=f"Doppler Velocity [{ds['W'].attrs['units']}]", cbar_pos=[cbar_left, cbar_top -6 * cbar_step, cbar_width, cbar_height])
    
#     format_range_axis(axs[3], ds)

    # # Reflectivity in dbz at 400 m
    # ze_400m = ds['Ze'].isel(range=3)#.dropna(dim='time')
    # axs[3].plot(ze_400m['time'].values, ze_400m.values, color='darkviolet', linestyle='-')#s=2.5)
    # axs[3].set_ylabel(f"Reflectivity [{ds['Ze'].attrs['units']}]")
    # axs[3].set_yticks(np.arange(-10,50,10))
    # axs[3].grid(True, ls=':')

    # # Reflectivity in linear units mm^6 m^-3
    # ze_lin_400m = ds['Ze_linear'].isel(range=3)
    # axs[4].plot(ze_lin_400m['time'].values, ze_lin_400m.values, color='darkorange', linestyle='-')
    # axs[4].set_ylabel("Ze [mm${^6}$/m${^3}$]")
    # axs[4].grid(True, ls=':')
    # axs[4].set_yscale('log')


#     # # ceilometer backscatter   
#     # bsc = ds['bsc'].dropna(dim=('altitude'), how='all')/1e-6   # Keep rows with at least 1 valid value along the 'altitude' dimension
#     # bsc = bsc.transpose('altitude', 'time')
#     # cbh = ds['cbh'].isel(layer = 1).where(ds['cbh'].isel(layer = 1) > -1.)
#     # plot_pcolormesh_with_cbar(ax=axs[0], fig=fig, data=bsc, x=time_vals, y=ds['altitude'].where(~np.isnan(bsc)), cmap=cmap, norm=norm, cbar_bounds=bounds, cbar_label=f"Backscatter [{ds['bsc'].attrs['units']}]", cbar_pos=[cbar_left, cbar_top - 0 * cbar_step, cbar_width, cbar_height])

#     # plot_pcolormesh_with_cbar(ax=axs[0], fig=fig, data=(ds['bsc']).dropna('altitude').transpose('altitude', 'time'), x=time_vals, y=ds['altitude'], cmap=cmap, norm=norm, cbar_bounds=np.arange(0,1.2e8,0.2e8), cbar_label='Backscatter', cbar_pos=[cbar_left, cbar_top - 0 * cbar_step, cbar_width, cbar_height])
    
#     # plot cloud base every 5 minutes
#     if pd.Timestamp("2023-02-23") <= day_start <= pd.Timestamp("2023-03-13"):
#         cbh_data = ds['cbh'].isel(layer=1).resample(time='5min').nearest()
#     else:
#         cbh_data = ds['cbh'].isel(layer=0).resample(time='5min').nearest()
#     # cbh_data = ds['cbh'].isel(layer=0).resample(time='5min').nearest()
#     cbh_attrs = ds['cbh'].attrs['units']
#     axs[0].plot(date2num(cbh_data['time']), cbh_data.values, ".",linestyle = '-', markersize=2, alpha=0.3, color='grey', label=f"Cloud Base [{cbh_attrs}]")
#     # axs[0].scatter(cbh_data['time'].values, cbh_data.values, marker=".", s=1.5, alpha=0.3, color='grey', label="Cloud Base")

#     axs[0].set_yticks(np.arange(0, 5001, 1000))
#     axs[0].set_ylim(0, 5000)
#     axs[0].set_ylabel("Height [m]")
#     axs[0].legend()
#     
#     

    
#     # spectral Width
#     bounds  = np.arange(0,1.1,0.1)
#     norm    = BoundaryNorm(boundaries=bounds, ncolors=256)
#     cmap    = cm.batlow_r
    
# #     sWidth  = axs[3].pcolormesh(time_vals, ds['range'], ds['spectralWidth'].transpose('range', 'time'), cmap=cmap, norm=norm, shading='auto')
#     plot_pcolormesh_with_cbar(ax=axs[3], fig=fig, data=ds['spectralWidth'].transpose('range', 'time'), x=time_vals, y=ds['range'], cmap=cmap, norm=norm, cbar_bounds=np.arange(0,1.2,0.2), cbar_label=f"Spectral Width [{ds['spectralWidth'].attrs['units']}]", cbar_pos=[cbar_left, cbar_top -3 * cbar_step, cbar_width, cbar_height])
    
#     format_range_axis(axs[3], ds)

    
    
    
#     # 2m Temperature
#     temp_data = ds['air_temperature'].dropna(dim='time')#.resample(time='5min').nearest()
#     axs[5].plot(temp_data['time'].values, temp_data.values, color='tab:red')

#     axs[5].hlines(y=2., xmin=day_start, xmax=xmax, color='grey')
#     axs[5].set_ylabel(f"2m temperature [{ds['air_temperature'].attrs['units']}]")
#     axs[5].set_ylim([-11.5, 12.5])


    
#     # 10m wind
#     wd_data = ds['wind_direction'].dropna(dim='time').resample(time='5min').nearest()
#     axs[6].scatter(wd_data['time'].values, wd_data.values, color='tab:blue', s=2.5)
#     axs[6].set_ylabel(f"Wind direction [{ds['wind_direction'].attrs['units']}]", color='tab:blue')
#     axs[6].tick_params(axis='y', labelcolor='tab:blue')
#     axs[6].set_yticks(range(0, 360+90,90))
#     axs[6].set_ylim(0,360)
    
#     ax2 = axs[6].twinx()
#     ws_data = ds['wind_speed'].dropna(dim='time').resample(time='5min').nearest()
#     ax2.plot(ws_data['time'].values, ws_data.values, color='tab:green')
#     ax2.set_ylabel(f"Wind speed [{ds['wind_speed'].attrs['units']}]", color='tab:green')
#     ax2.tick_params(axis='y', labelcolor='tab:green')
#     ax2.set_ylim(0,20)

    

    

    # Shared formatting
    date_form = DateFormatter("%H:00")
    for (i,ax), title in zip(enumerate(axs), titles):
        ax.set(xlabel = "", title=title)
        ax.text(0.045, 0.9, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
        ax.xaxis.grid(ls=':')
        ax.tick_params(axis='x', labelrotation=0)
        plt.setp(ax.get_xticklabels(), horizontalalignment="center")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xlim([day_start, xmax])

    axs[-1].set(xlabel="Time [UTC]")
    fig.suptitle(f"{site.split('.')[0]} ({ds.time.dt.date[0].values} — {ds.time.dt.date[-1].values})", fontweight="bold")
    # fig.subplots_adjust(top=0.94)
    fig.subplots_adjust(top=0.91)

def test_and_return_dominant_sf_class(seg, sclass, threshold):
    # count the occurences of the snowflake class name within the dTime
    aggregate_count = seg.str.count('aggregate').sum()
    graupel_count = seg.str.count('graupel').sum()
    pristine_count = seg.str.count('pristine').sum()
    total_count = len(seg)

    # calculate the fraction of this number to find which snowflake class dominates within dTime
    frac_agg = aggregate_count/total_count
    frac_graupel = graupel_count/total_count
    frac_pris = pristine_count/total_count

    largest_number = max(frac_agg, frac_graupel, frac_pris)

    # assign dominant snowflake class to dTime if it occurs more than 60% 
    if largest_number >= threshold:
        if largest_number == frac_agg:
            sclass.append(f'aggregate')
        elif largest_number == frac_graupel:
            sclass.append(f'graupel')
        elif largest_number == frac_pris:
            sclass.append(f'pristine')
    if largest_number < threshold:
        sclass.append(f'mixed')
    return sclass

def model(x, a, b):
    return a * x**b
    # return (x/a)**(1/b)
    

def process_precip_segments(df_sum, df_mean, sf_class, mode="event", duration="1H", min_rate=0.1, altitudes=[400, 800, 1000], a=56, b=1.2):
    if mode == "event":
        notna_idx = df_sum['precipitation'].dropna().index
        notna_idx = notna_idx.append(pd.Index([df_sum.index[-1]]))
        segment_bounds = notna_idx

    elif mode == "fixed":
        step = pd.to_timedelta(duration)
        step = int(step/pd.Timedelta(seconds=60))
        dt_idx = df_sum.index[step::step] # step in minutes
        dt_idx = dt_idx.append(pd.Index([df_sum.index[-1] + pd.Timedelta(minutes=1)]))
        segment_bounds = dt_idx
    else:
        raise ValueError("mode must be 'event' or 'fixed'")
    
    # Loop over segments
    sums, means, sclass_60, sclass_70, sclass_80, mascdb, counts, zs_counts = [], [], [], [], [], [], [], []
    dt = segment_bounds.to_series().diff().dt.total_seconds()
    for i in range(len(segment_bounds)):
        if i == 0:
            seg = df_sum.loc[:segment_bounds[0]].copy()
            seg_m = df_mean.loc[:segment_bounds[0]].copy()
            seg_sf = sf_class.loc[:segment_bounds[0]].copy()
            delta_h = dt.loc[:segment_bounds[0]]

        else:
            seg = df_sum.loc[segment_bounds[i-1] + pd.Timedelta(seconds=1) : segment_bounds[i]].copy()
            seg_m = df_mean.loc[segment_bounds[i-1] + pd.Timedelta(seconds=1) : segment_bounds[i]].copy()
            seg_sf = sf_class.loc[segment_bounds[i-1] + pd.Timedelta(seconds=1) : segment_bounds[i]].copy()
            delta_h = dt.loc[segment_bounds[i-1] + pd.Timedelta(seconds=1) : segment_bounds[i]].copy()

        # filter: remove warm segments (>2 °C)
        all_T = seg_m['air_temperature'].dropna()
        if (len(all_T) != 0) and (sum(all_T > 2.) >=1):
            seg.loc[:,:] = np.nan
            seg_m.loc[:,:] = np.nan
            seg_sf.loc[:,:] = np.nan

        sums.append(seg.sum(numeric_only=True, min_count=1))
        means.append(seg_m.mean(numeric_only=True, skipna=True))
        mascdb.append(seg_sf[['fallspeed', 'flake_Dmax', 'gan3d_mass']].mean(numeric_only=True, skipna=True))
        ma_cc = seg_sf[['flake_Dmax', 'fallspeed', 'gan3d_mass']].count()
        ma_cc = ma_cc.rename({"flake_Dmax":"count_flake_Dmax", "fallspeed":"count_fallspeed", "gan3d_mass":"count_gan3d_mass"})
        counts.append(ma_cc)
        zs_cc = seg[['precipitation'#,'Ze_linear_sum_400', 'Ze_linear_sum_800', 'Ze_linear_sum_1000'
        ]].count()
        zs_cc = zs_cc.rename({"precipitation":"count_precip", #"Ze_linear_sum_400":"count_Ze_lin_400", "Ze_linear_sum_800":"count_Ze_lin_800", "Ze_linear_sum_1000":"count_Ze_lin_1000"
        #                       
        })
        zs_counts.append(zs_cc)
        

        if len(seg_sf) != 0:
            sclass_60 = test_and_return_dominant_sf_class(seg_sf['snowflake_class_name'], sclass_60, 0.60)
            sclass_70 = test_and_return_dominant_sf_class(seg_sf['snowflake_class_name'], sclass_70, 0.70)
            sclass_80 = test_and_return_dominant_sf_class(seg_sf['snowflake_class_name'], sclass_80, 0.80)
            if i== 0:
                psd = find_psd_in_timestep(seg_sf, delta_h, index=segment_bounds[i])
            else:
                ds_psd = find_psd_in_timestep(seg_sf, delta_h, index=segment_bounds[i])
                psd = xr.concat([psd, ds_psd], dim="time")

        else:
            sclass_60.append(np.nan)
            sclass_70.append(np.nan)
            sclass_80.append(np.nan)
            if i== 0:
                psd = find_psd_in_timestep(seg_sf, delta_h, index=segment_bounds[i])
            else:
                ds_psd = find_psd_in_timestep(seg_sf, delta_h, index=segment_bounds[i])
                psd = xr.concat([psd, ds_psd], dim="time")
            

        
    # Combine into one Dataframe with all values
    df_out = pd.concat([pd.DataFrame(sums).set_index(segment_bounds),
                        pd.DataFrame(means).set_index(segment_bounds),
                        pd.DataFrame({"snowflake_class_name_60":sclass_60}).set_index(segment_bounds), 
                        pd.DataFrame({"snowflake_class_name_70":sclass_70}).set_index(segment_bounds),
                        pd.DataFrame({"snowflake_class_name_80":sclass_80}).set_index(segment_bounds),
                        pd.DataFrame(mascdb).set_index(segment_bounds),
                        pd.DataFrame(counts).set_index(segment_bounds),
                        pd.DataFrame(zs_counts).set_index(segment_bounds)
                        # pd.DataFrame({"Dmax":dm}).set_index(segment_bounds),
                        # pd.DataFrame({"particle_mass":mass}).set_index(segment_bounds),
                        ], axis=1)
    df_out.rename(columns={"precipitation":"Geonor",}, inplace=True)

    # Convert snowfall increments to rate (mm/h)
    df_out["delta_h"] = df_out.index.to_series().diff().dt.total_seconds() / 3600 # Calculate time difference in hours
    df_out["rate_mm_h"] = df_out["Geonor"] / df_out["delta_h"] # Convert to mm/h rate
    df_out[df_out["rate_mm_h"] < min_rate] = np.nan # limit Geonor precipitation to be above 0.1 mm/h
    df_out['start_time'] = df_out.index - pd.to_timedelta(df_out['delta_h'] * 3600, unit='sec')

    # for alt in altitudes:
        # Z from linear sums/means
        # df_out[f'Z_from_linear_sum_{alt}'] = df_out[f'Ze_linear_sum_{alt}'].apply(lambda y: 10 * np.log10(y) if y > 0 else np.nan)
        # df_out[f'Z_from_linear_mean_{alt}'] = df_out[f'Ze_linear_mean_{alt}'].apply(lambda y: 10 * np.log10(y) if y > 0 else np.nan)
        # Rates
    alt = "400"
    for name in ["Matrosov_aggregate", "Bracci_aggregate", "Nakai_aggregate", "Nakai_graupel", "Pu_graupel", "Schoger", f"S_{a}_{b}"]:
        df_out[f"{name}_{alt}_mm_h"] = df_out[f"{name}_{alt}m"] / df_out["delta_h"]

        # # Matrosov from Ze_sum
        # df_out[f'Matrosov_from_ze_sum_{alt}'] = ((df_out[f'Ze_linear_sum_{alt}'] / 56) ** (1/1.2)) /(60*df_out["delta_h"])  # convert rate to mm

    df_out.index.name = 'time'
    # ds_out = xr.concat([df_out.to_xarray(), psd], "time")
    
    # in addition in the dataframe fallspeed has to be present to adjust for the timing for the reflectivity
    df_out = df_out[df_out['fallspeed'].notna()]
    df_out.index

    psd = psd.sel(time=df_out.index)

    return df_out, psd

def find_psd_in_timestep(seg_sf, delta_h, index):
    # Define diameter bins. Choose logarithmic (geometric) bins in diameter 
    bin_edges = np.geomspace(0.0003, 0.03, 25) # Return numbers spaced evenly on a log scale (a geometric progression). 25 bins
    dD = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + dD/2

    # Assign particles to bins
    bin_idx = np.digitize(seg_sf['flake_Dmax'], bin_edges) - 1 # Return the indices of the bins to which each value in input array belongs

    # Compute mean v and m per diameter bin
    velocity = seg_sf['fallspeed'].values
    valid = ~np.isnan(velocity)
    v_bin = np.array([np.nanmean(velocity[(bin_idx==i) & valid]) if np.any((bin_idx==i) & valid) else np.nan for i in range(len(bin_centers)) ]) # fall speed per bin [m/s]
    # m_bin = np.array([np.nanmean(seg_sf['gan3d_mass'][bin_idx==i]) if np.any(bin_idx==i) else np.nan for i in range(len(bin_centers))]) 
    mass = seg_sf['gan3d_mass'].values
    valid = ~np.isnan(mass)
    m_bin = np.array([np.nanmean(mass[(bin_idx == i) & valid]) if np.any((bin_idx == i) & valid) else np.nan for i in range(len(bin_centers))])# mass per particle for each bin

    # count particles per diameter bin
    counts, _ = np.histogram(seg_sf['flake_Dmax'], bins=bin_edges)


    # Convert counts to # concentration
    # effective area
    A_eff = 3.1e-3 # trigger cross-section (m^2). default 3100 mm^2
    T_sample = delta_h # seconds


    # number concentration per bin (particles m^-3)
    N_bin = np.full_like(counts, np.nan, dtype=float)
    valid = (v_bin > 0) 
    N_bin[valid] = counts[valid] / (v_bin[valid] * A_eff * T_sample.values)

    # density per unit diameter (particles m^-3 per m)
    n_per_m = N_bin / dD # per meter

    # # Compute snowfall rate in that delta_h
    # rho_w = 1000. # liquid water density [kg/m^3]
    # if np.all(np.isnan(m_bin * v_bin * N_bin * dD)):
    #     S_m_per_h = np.nan
    # else:
    #     # S_m_per_h = (3600.0 / rho_w) * np.nansum(m_bin * v_bin * N_bin * dD) # mm/h
    #     S_m_per_h = (3600. / rho_w) * (np.nansum(seg_sf['gan3d_mass'] * seg_sf['fallspeed']) / (A_eff * T_sample))

    # df_psd = pd.DataFrame({'D_center_m': bin_centers, 'dD_m': dD, 'counts':counts, 'v_bin_m_s':v_bin, 'N_bin_m3':N_bin, 'n_per_m':n_per_m,})
    dataset = xr.Dataset({"counts": (["time", "bin"], [counts]), 
                        # "bin_edges": (["time", "bin"],[ bin_edges]),
                          "v_bin_m_s": (["time", "bin"], [v_bin]),
                          "m_bin_kg":(["time","bin"], [m_bin]),
                        #   "D_center_m": (["time", "bin"], [dD]),
                          "N_bin_m3": (["time", "bin"], [N_bin]),
                          "n_per_m": (["time", "bin"], [n_per_m]),
                        #   "S_m_per_h": (["time"], [S_m_per_h]),
                          }, 
                          coords={"time": [index], 
                                  "bin": bin_centers,
                                  "dD":("bin", dD) # widths as coordinate
                                  })
    return dataset



def plt_geonor_vs_ze_vs_precip_scatter(ylabels, class_based=False,  df=None, df_only_class=None, xlim=(0.09, 30), ylim_panel0=None, ylim_panel1=None, title=None):

    """
    Generalized routine to plot two scatter panels:
    - Panel 0: scatter against Ze (with Matrosov line)
    - Panel 1: scatter against Matrosov (with 1-1 line)

    Parameters
    ----------
    df : pd.DataFrame
        Data for scatter plots (when not class_based).
    ylabels : list[str]
        Axis labels for the two panels.
    xlim, ylim_panel0, ylim_panel1 : tuple or None
        Axis limits.
    class_based : bool
        If True, use df_only_class
    df_only_class : dict[str, pd.DataFrame]
        Class-split data for class_based mode.
    """


    fig, axsm = plt.subplots(1, 2, figsize=(10, 5, ), sharex=True, )
    axs = axsm.flatten()

    if not class_based:
        # scatter by altitude
        for i, alt in enumerate([1000, 800, 400]):
            axs[0].scatter(df['rate_mm_h'], df[f'Ze_linear_sum_{alt}'], color=cmap(i), alpha=0.7, label=f'{alt} m', edgecolor=cmap(i))
            axs[1].scatter(df['rate_mm_h'], df[f'Matrosov_a_{alt}_mm_h'], color=cmap(i), alpha=0.7, label=f'{alt} m', edgecolor=cmap(i))
    else:
        # scatter by snowflake class
        for (class_name, color) in custom_colors.items():
            axs[0].scatter(df_only_class[class_name]['rate_mm_h'], df_only_class[class_name]['Ze_linear_sum_400'], color=color, alpha=0.7, label=f"{class_name}".title(), edgecolor=color)
            axs[1].scatter(df_only_class[class_name]['rate_mm_h'], df_only_class[class_name][f'Matrosov_a_400_mm_h'], color=color, alpha=0.7, label=f"{class_name}".title(), edgecolor=color)
        
    # Shared formatting
    # Generate line for Matrosov
    # x_line = np.linspace(0, 50, 200)
    x_line = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 300)

    y_line = 56 * x_line**1.2

    # 1-1 line
    x = np.linspace(0, 50, 100)
    y = x

    bb = (fig.subplotpars.left, fig.subplotpars.bottom-0.21, 
        fig.subplotpars.right-fig.subplotpars.left,.1)
    
    for i, ax in enumerate(axs.flatten()):
        ax.grid(True, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.text(0.045, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
        if i == 0: 
            ax.plot(x_line, y_line, color='black', label="Matrosov$_{a}$: Z = 56 S$^{1.2}$", linestyle='solid')
            ax.plot(x_line, 134 * x_line**1.25, color='black', label = "Bracci$_{a}$: Z = 134 S$^{1.25}$", linestyle = 'dotted')
            ax.plot(x_line, 107.4 * x_line**1.67, color='black', label = "Nakai$_{a}$: Z = 107.4 S$^{1.67}$", linestyle = 'dashed')
            ax.plot(x_line, 227.4 * x_line**1.67, color='black', label = "Nakai$_{g}$: Z = 227.4 S$^{1.67}$", linestyle = 'dashdot')
            ax.plot(x_line, 252 * x_line**2.03, color='black', label = "Pu$_{g}$: Z = 252 S$^{2.03}$", linestyle = (0, (3, 1, 1, 1)))
            ax.set(xlabel='Geonor [mm h$^{-1}$]', ylabel=ylabels[0], xlim=xlim, )
            if ylim_panel0: ax.set_ylim(ylim_panel0)
            ax.legend(bbox_to_anchor=bb, mode="expand", loc="lower left",
                ncol=3, borderaxespad=0., bbox_transform=fig.transFigure)

        else:
            ax.plot(x, y, linestyle='--', color='darkgrey')
            ax.set(xlabel='Geonor [mm h$^{-1}$]', ylabel=ylabels[1], xlim=xlim, ylim=(0.09, 50),  )
            if ylim_panel1: ax.set_ylim(ylim_panel1)

    fig.suptitle(f"{title}", fontweight ="bold",y=0.92)  
    plt.tight_layout(rect=[0,0,1,1])


def plt_masc_vs_mrr_fallspeed(
    class_based=False,
    df=None,
    df_only_class=None,
    panel_keys=None,
    altitude_for_classes=400,
    xlim=(0.0, 7.0),
    ylim=(0.0, 7.0),
    title=None
):
    """
    Plot one panel per panel_key.

    - If class_based==False:
        panel_keys should be a list of altitudes (ints). Each panel shows df['fallspeed'] vs df[f'W_{alt}m'].
        Default: [1000, 800, 400]

    - If class_based==True:
        panel_keys should be a list of class names. Each panel shows that class at altitude_for_classes:
            df_only_class[class_name]['fallspeed'] vs df_only_class[class_name][f'W_{altitude_for_classes}m']
        Default: first 3 keys from fct.custom_colors.

    Returns (fig, axs).
    """
    # decide panel keys
    if class_based:
        if panel_keys is None:
            panel_keys = list(custom_colors.keys())[:3]
    else:
        if panel_keys is None:
            panel_keys = [1000, 800, 400]

    n_panels = len(panel_keys)
    fig, axsm = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharex=True, sharey=True)

    # normalize axs to iterable list
    if n_panels == 1:
        axs = [axsm]
    else:
        axs = axsm.flatten()

    # 1:1 line
    x = np.linspace(0, 7, 100)
    y = x

    for i, (ax, key) in enumerate(zip(axs, panel_keys)):
        ax.plot(x, y, linestyle='--', color='darkgrey')  # 1:1
        ax.grid(True, ls=':')

        # choose series depending on mode
        if not class_based:
            alt = key
            x_series = df['fallspeed']
            y_series = df[f'W_{alt}m']
            color = cmap(i)
            label = f"{alt} m"
            ax.scatter(x_series, y_series, color=color, alpha=0.7,
                       edgecolor=color, label=label)
            masc_mean = x_series.mean(skipna=True)
            mrr_mean = y_series.mean(skipna=True)
            tot = int(y_series.dropna().shape[0])
            ax.set_title(f"N={tot} @{alt} m", color=color, fontweight="bold")

        else:
            class_name = key
            # guard against missing class
            if (df_only_class is None) or (class_name not in df_only_class):
                masc_mean = np.nan
                mrr_mean = np.nan
                tot = 0
                ax.text(0.5, 0.5, f"No data for\n{class_name}", ha='center', va='center', transform=ax.transAxes)
                color = cmap(i)
            else:
                x_series = df_only_class[class_name]['fallspeed']
                y_series = df_only_class[class_name][f'W_{altitude_for_classes}m']
                color = custom_colors.get(class_name, cmap(i))
                ax.scatter(x_series, y_series, color=color, alpha=0.7, edgecolor=color, label=class_name)
                masc_mean = x_series.mean(skipna=True)
                mrr_mean = y_series.mean(skipna=True)
                tot = int(y_series.dropna().shape[0])
                ax.set_title(f"{class_name}: N={tot}".title(), color=color, fontweight="bold")

        # small stats box (handles no-data gracefully)
        if np.isfinite(masc_mean) and np.isfinite(mrr_mean):
            stats_text = (f"MASC$_{{mean}}$: {masc_mean:.2f} m s$^{{-1}}$\n"
                          f"MRR$_{{mean}}$: {mrr_mean:.2f} m s$^{{-1}}$")
        else:
            stats_text = "No data"

        ax.text(0.7, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='k', lw=1, alpha=0.8))

        ax.set_xlabel(r"Fallspeed - MASC [m s$^{-1}$]")
        if i == 0:
            ax.set_ylabel(r"Fallspeed - MRR [m s$^{-1}$]")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # small panel label if available
        if i < len(fig_label):
            ax.text(0.05, 0.95, fig_label[i], fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes)
    fig.suptitle(f"{title}", fontweight ="bold",y=0.92)  
    plt.tight_layout(rect=[0,0,1,1])

def get_ds_in_precip_event(df, ds):

    slices = []
    for end_time, start_time in df['start_time'].items():
        t0 = start_time #+ pd.Timedelta(seconds=1)
        t1 = end_time-pd.Timedelta(seconds=1)
        # Select data from ds 
        # ds_subset = ds[['Ze', 'height']].sel(time=slice(t0, t1))
        ds_subset = ds.sel(time=slice(t0, t1))

        # # Add a new "interval" dimension with the end_time as its label
        # ds_subset = ds_subset.expand_dims(interval=[pd.to_datetime(end_time)])

        slices.append(ds_subset)

    result = xr.concat(slices, dim='time').sortby("time")

    return result

def plt_fallspeed_distribution_by_class(df_only_class, binwidth=0.25,stat='percent', binrange_v=[0,7], binrange_m=[0, 0.0525], title=None):

    fig, axsm = plt.subplots(2, 3, figsize=(15, 10, ), sharex=False, sharey=False)
    axs = axsm.flatten()

    for (i,ax), (class_name) in zip(enumerate(axs), ['aggregate', 'graupel', 'mixed']):
        sns.histplot(df_only_class[class_name]['fallspeed'], stat=stat, binwidth=binwidth, binrange=binrange_v, ax=ax, color=custom_colors[class_name], alpha=0.7, edgecolor=custom_colors[class_name] )
        sns.histplot(df_only_class[class_name]['gan3d_mass']/1e-3, stat='percent', binwidth=0.0025, binrange=binrange_m, ax=axsm[1,i], color=custom_colors[class_name], alpha=0.7, edgecolor=custom_colors[class_name] )
        
        
        
        
        ax.set_title(f"{class_name}".title(), color=custom_colors[class_name], fontweight='bold', )

    for i,ax in enumerate(axs): 
        ax.grid(True, ls=':')
        ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
        if i == 0 or i == 1 or i==2:
            ax.set(xlabel = "Fallspeed [m s$^{-1}$]", ylim=[0, 35])
        else:
            ax.set(xlabel = "Particle mass [g]", ylim=[0,35])
    fig.suptitle(f"{title}", fontweight ="bold",y=0.98)  
    plt.tight_layout(rect=[0,0,1,1])

    # plt.savefig(os.path.join(FIG_DIR, 'fallspeed', 'aggregate_graupel_mixed_pristine_precip_event.png'), format='png', bbox_inches='tight', dpi=300, transparent=True,)

def plt_prob_agg_graupel_mixed(H_ze_prob, H_ze, H_w_prob, H_w, H_ze_w_prob, H_ze_w, category):
    cmap = cm.managua_r #cm.glasgow_r  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be white
    cmaplist[0] = (255,255,255, 0.9)
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 40) # cmap.N
    # define the bins and normalize
    # bounds = np.linspace(0, 7e7, 11)
    # bounds = np.arange(0., 0.003, 0.0002)
    # bounds_z = np.round(np.linspace(0., 0.004, 15), 4)
    bounds_z = np.arange(0., 0.004, 0.0001)
    norm_z = BoundaryNorm(bounds_z, cmap.N)

    # bounds_w = np.round(np.linspace(0., 0.0025, 15), 4)
    bounds_w = bounds_z
    norm_w = BoundaryNorm(bounds_w, cmap.N)

    bounds_z_w = bounds_z
    norm_z_w = BoundaryNorm(bounds_z_w, cmap.N)

    fig, axs = plt.subplots(
                nrows=3, ncols=3,
                figsize=(15,15), )#sharey=True, 
                
    axs = np.atleast_2d(axs)    # safe even if n_cols=2

    categories = category*3
    for (i,ax), cat in zip(enumerate(axs.flatten()), categories):
                threshold = cat.split("_")
                if len(threshold) > 1:
                    threshold = threshold[1]
                    cate = cat.replace(f"_{threshold}", "")
                else: 
                    cate = cat
                if i in [0, 1, 2]:
                    # pcm = ax.pcolormesh(ze_bins, height_bins, H_ze_prob[cat], cmap=cmap, norm=norm_z) 
                    # pcm = ax.pcolormesh(np.arange(-10, 31, 1), np.arange(100, 3100,100), H_ze_prob[cat], cmap=cmap, norm=norm_z) #,levels=bounds_z,
                    pcm = ax.contourf(np.arange(-10, 31, 1), np.arange(100, 3100,100), H_ze_prob[cat], 
                                      levels=bounds_z, 
                                      cmap=cmap, norm=norm_z, extend="max")
                    


                    # title with raw counts
                    total = int(np.nansum(H_ze[cat]))
                    
                    ax.set_title(f"{cate}\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
                    ax.set_xlabel("Reflectivity [dBz]")
                elif i in [3, 4, 5]:
                    # pcm2 = ax.pcolormesh(np.arange(-4, 4.1, 0.1), np.arange(100, 3100,100), H_w_prob[cat], cmap=cmap, norm=norm_w) 
                    pcm2 = ax.contourf(np.arange(-4, 4.1, 0.1), np.arange(100, 3100, 100), H_w_prob[cat], levels=bounds_w, cmap=cmap, norm=norm_w, extend="max")
                    ax.vlines(0,0,3000, linestyle='--', color='darkgrey')
                    total = int(np.nansum(H_w[cat]))
                    ax.set_title(f"\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
                    ax.set_xlabel("Doppler Velocity [m s$^{{-1}}$]")

                elif i in [6, 7, 8]:
                    # Plot
                    pcm3 = ax.contourf(np.arange(-4, 4.1, 0.1), np.arange(-10, 31, 1), H_ze_w_prob[cat], levels=bounds_z_w, cmap=cmap, norm=norm_z_w, extend="max")
                    total = int(np.nansum(H_ze_w[cat]))
                    ax.set_title(f"\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
                    ax.vlines(0,-10,30, linestyle='--', color='darkgrey')
                    ax.set_ylabel("Reflectivity [dBz]")
                    ax.set_xlabel("Doppler Velocity [m s$^{{-1}}$]")
            

                ax.grid(True, ls=':')
                ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
                if i in [0, 3]:
                    ax.set_ylim([0, 3000])
                    ax.set_ylabel("Height [m]")
                


    plt.tight_layout(rect=[0, 0, 0.9, 1])   # leave room for cbars

    cax = fig.add_axes([0.89, 0.71 , 0.015, 0.24]) # left, bottom, width, height
    cbar = fig.colorbar(pcm, ticks=bounds_z[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()
    # move the exponent text to be centered above bar
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)

    cax = fig.add_axes([0.89, 0.38 , 0.015, 0.24]) # left, bottom, width, height
    cbar2 = fig.colorbar(pcm2, ticks=bounds_w[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar2.formatter = formatter
    cbar2.update_ticks()
    # move the exponent text to be centered above bar
    offset = cbar2.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)

    cax = fig.add_axes([0.89, 0.05 , 0.015, 0.24]) # left, bottom, width, height
    cbar3 = fig.colorbar(pcm3, ticks=bounds_z_w[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar3.formatter = formatter
    cbar3.update_ticks()
    # move the exponent text to be centered above bar
    offset = cbar3.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)

def plt_prob_agg_graupel_Ze_W(H_ze_prob, H_ze, H_w_prob, H_w, category):
    cmap = cm.managua_r #cm.glasgow_r  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be white
    cmaplist[0] = (255,255,255, 0.9)
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 40) # cmap.N
    bounds_z = np.arange(0., 0.004, 0.0001)
    norm_z = BoundaryNorm(bounds_z, cmap.N)

    bounds_w = bounds_z
    norm_w = BoundaryNorm(bounds_w, cmap.N)


    fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(15,15), )#sharey=True, 
                
    axs = np.atleast_2d(axs)    # safe even if n_cols=2

    categories = category*3
    for (i,ax), cat in zip(enumerate(axs.flatten()), categories):
                threshold = cat.split("_")
                if len(threshold) > 1:
                    threshold = threshold[1]
                    cate = cat.replace(f"_{threshold}", "")
                else: 
                    cate = cat
                if i in [0, 1]:
                    pcm = ax.contourf(np.arange(-10, 31, 1), np.arange(100, 3100,100), H_ze_prob[cat], 
                                      levels=bounds_z, 
                                      cmap=cmap, norm=norm_z, extend="max")
                    


                    # title with raw counts
                    total = int(np.nansum(H_ze[cat]))
                    
                    ax.set_title(f"{cate}\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
                    ax.set_xlabel("Reflectivity [dBZ]")
                elif i in [2, 3]:
                    pcm2 = ax.contourf(np.arange(-4, 4.1, 0.1), np.arange(100, 3100, 100), H_w_prob[cat], levels=bounds_w, cmap=cmap, norm=norm_w, extend="max")
                    ax.vlines(0,0,3000, linestyle='--', color='darkgrey')
                    total = int(np.nansum(H_w[cat]))
                    ax.set_title(f"\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
                    ax.set_xlabel("Doppler Velocity [m s$^{{-1}}$]")

            

                ax.grid(True, ls=':')
                ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
                if i in [0, 2]:
                    ax.set_ylim([0, 3000])
                    ax.set_ylabel("Height [m]")
                


    plt.tight_layout(rect=[0, 0, 0.9, 1])   # leave room for cbars

    cax = fig.add_axes([0.89, 0.58 , 0.015, 0.34]) # left, bottom, width, height
    cbar = fig.colorbar(pcm, ticks=bounds_z[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()
    # move the exponent text to be centered above bar
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)

    cax = fig.add_axes([0.89, 0.08 , 0.015, 0.34]) # left, bottom, width, height
    cbar2 = fig.colorbar(pcm2, ticks=bounds_w[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar2.formatter = formatter
    cbar2.update_ticks()
    # move the exponent text to be centered above bar
    offset = cbar2.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)

    

def plt_CFAD_agg_graupel_mixed_all(H_ze_prob, H_ze, x_range, category, xlabel):
    cmap = cm.managua_r #cm.glasgow_r  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be white
    cmaplist[0] = (255,255,255, 0.9)
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 40) # cmap.N
    # define the bins and normalize
    bounds_z = np.arange(0., 0.004, 0.0001)
    norm_z = BoundaryNorm(bounds_z, cmap.N)
    

    fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(15,15), sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_2d(axs)
    for (i, ax), cat in zip(enumerate(axs.flatten()), category):
        threshold = cat.split("_")
        if len(threshold) > 1:
            threshold = threshold[1]
            cate = cat.replace(f"_{threshold}", "")
        else:
            cate = cat
        pcm = ax.contourf(x_range, np.arange(100, 3100, 100), H_ze_prob[cat], levels=bounds_z, cmap=cmap, norm=norm_z, extend="max") 
        # title with raw counts
        total = int(np.nansum(H_ze[cat]))
        ax.set_title(f"{cate}\n(counts={total})".title(), fontweight='bold', color=custom_colors[cate])
        ax.set_xlabel(xlabel)

        ax.grid(True, ls=':')
        ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
        if i in [0, 3]:
            ax.set_ylim([0, 3000])
            ax.set_ylabel("Height [m]")
    cax = fig.add_axes([1.01, 0.05 , 0.015, .9]) # left, bottom, width, height
    cbar = fig.colorbar(pcm, ticks=bounds_z[::4], cax=cax, orientation='vertical', label='Probability', extend='max', spacing="proportional")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()

    # move the exponent text to be centered above bar
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_horizontalalignment('center')
    offset.set_x(0.5)


def plt_precip_scatter_literature(df, title=" "):
    cmap = cm.managua  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    custom_cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 7) # cmap.N
    equations = {"Matrosov_a": ['Aggregate', "(2007)", "#C51B7D", 56, 1.2],#["Z = 56 S$^{1.2}$", ], 
             "Bracci_a": ['Aggregate', "et al. (2022)", "#F768A1", 134, 1.25],#["Z = 134 S$^{1.25}$", ],
            #  "Nakai_a": ['Aggregate', "et al. (2022)", custom_cmap(2)],#["Z = 107.4 S$^{1.67}$"],
            #  "Nakai_g": ['Graupel', "et al. (2022)", custom_cmap(4)], # ["Z = 227.4 S$^{1.67}$", ],
             "Pu_g" : ['Graupel', "et al. (2020)", "#FDB863", 252, 2.03], #["Z = 252 S$^{2.03}$", ]
             "Schoger": ['All', "et al. (2021)", "#00BFC4", 77.61, 1.22]
             }

    fig, axsm = plt.subplots(2, 2, figsize=(10, 10, ), sharex=True, sharey=True, constrained_layout=True)
    axs = axsm.flatten()

    for (i,ax), (name, eq) in zip(enumerate(axs), equations.items()):
        if i == 0 or i==1:
            # color = custom_colors['aggregate'] 
            xval = df['aggregate']['rate_mm_h']
            yval = df['aggregate'][f'{name}_400_mm_h']
            # ylabel = "SWE$_{aggregate}$ [mm h$^{-1}$]"
        elif i ==2:
            xval = df['graupel']['rate_mm_h']
            yval = df['graupel'][f'{name}_400_mm_h']
            # ylabel = "SWE$_{graupel}$ [mm h$^{-1}$]"
        else:
            # color = custom_colors['graupel'] 
            xval = df['campaign']['rate_mm_h']
            yval = df['campaign'][f'{name}_400_mm_h']
            # ylabel = "SWE$_{all}$ [mm h$^{-1}$]"
        ax.scatter(xval, yval, color=eq[2], alpha=0.7,  edgecolor=eq[2])


        ax.set(xlim=[0.009, 40], ylim=[0.009,40])
        ax.grid(True, ls=':', which='both')
        ax.set_title(f"{name.split('_')[0]} {eq[1]}: {eq[0]}", color=eq[2], fontweight='bold', )

        ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes, color='black')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 1-1 line
        x = np.linspace(0, 40, 100)
        y = x
        ax.plot(x, y, linestyle='--', color='darkgrey')
        
        # annotation box
        text_box = ax.text(0.12, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

        mask = ((xval > 0) & (yval > 0) & np.isfinite(xval) & np.isfinite(yval))
        xval = xval[mask]
        yval = yval[mask]

        # Transform the data to logarithmic scales
        log_x = np.log10(xval)
        log_y = np.log10(yval)

        # compute error to the 1:1 line
        err = log_y - log_x
        rmse = np.sqrt(np.nanmean(err**2))

        # update annotation
        text_box.set_text(
                f"a = {eq[3]}\n"
                f"b = {eq[4]:.2f}\n"
                f"RMSE = {rmse:.4f}"
            )

    for i in range(2):
        axsm[i, 0].set_ylabel("SWE$_{retrieved}$ [mm h$^{-1}$]")
        axsm[1, i].set_xlabel("SWE$_{obs}$ [mm h$^{-1}$]")
    # axs[-2].text(2.05, 0.95, "\n Bracci et al. (2022) \n Matrosov (2007) \n Nakai et al. (2022) \n Pu et al. (2020)", fontweight ='bold', horizontalalignment='left',verticalalignment='center', transform = axs[-2].transAxes)
    # axs[-1].set_visible(False)
    # plt.rcParams['axes.labelcolor'] = 'white'
    # plt.rcParams['xtick.color'] = 'white'
    # plt.rcParams['ytick.color'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'white' # If using dark background

    fig.suptitle(f"{title}", fontweight ="bold",y=0.96)  
    # plt.tight_layout(rect=[0,0,1,1])

# Local slope (exponent b(D)) from quadratic fit
def local_exponent(D, coeffs):
    c2, c1, c0 = coeffs
    return c1 + 2 * c2 * np.log10(D)

def find_dimension_fits(df, variable):
    D = df['flake_Dmax']
    m = df[variable] # mean variable value per diameter bin


    if 'count_flake_Dmax' in df.columns:
        counts = (df[['count_flake_Dmax', f'count_{variable}']])
        counts = counts.min(axis=1, skipna=True, numeric_only=True)
        valid = (D > 0) & (m > 0) & (counts > 0)
        D, m, counts = D[valid], m[valid], counts[valid]
        # normalize counts to prevent huge differences
        weights = counts/counts.max()
    else:
        valid = (D > 0) & (m > 0)
        D, m = D[valid], m[valid]
        weights = np.ones(len(D))


    def residuals(params, x, y, w):
        a, b = params
        r = np.log10(a) + b * x -y
        return np.sqrt(w)*r # weighted residuals

    if len(D) != 0:
        logD = np.log10(D)
        logm = np.log10(m)

    

        p0 = (5, 5)
        res = least_squares(residuals,p0, args=(logD, logm, weights), bounds=([0,0], [np.inf, np.inf]))

        # variable-dimension fit
        # linear fit (deg=1)
        # lin_coeff = np.polyfit(logD, logm, 1) # slope, intercept
        # b_lin, a_lin = lin_coeff
        # log_lin_fit = np.polyval(lin_coeff, logD_sorted)
        a_lin, b_lin = res.x
        log_lin_fit = np.log10(a_lin * (D**b_lin))

        # sort for plotting
        idx = np.argsort(logD)
        logD_sorted = logD.iloc[idx]

        
        # sorted linear fit 
        log_lin_fit = log_lin_fit.iloc[idx] # ensure same order as logD_sorted
        # quadratic fit (deg=2)
        quad_coeff = np.polyfit(logD, logm, 2, w=weights) # c2, c1, c0
        c2, c1, c0 = quad_coeff
        log_quad_fit = np.polyval(quad_coeff, logD_sorted)

        b_local = local_exponent(D, quad_coeff)
    
    return logD_sorted, log_lin_fit, log_quad_fit, b_local, b_lin

def plt_dimension_fits(df, variable, logD, log_lin_fit, log_quad_fit, b_local, b_lin, var_name, ylabel, title=" "):
    D= df['flake_Dmax']#.values # bin centers
    m = df[variable]#.values # mean variable value per diameter bin
    # sort by diameter for plotting
    
    if 'count_flake_Dmax' in df.columns:
        counts = (df[['count_flake_Dmax', f'count_{variable}']])
        counts = counts.min(axis=1, skipna=True, numeric_only=True) # find the minimum amount of flakes within time bin of the two given values
        counts = counts.values
    else:
        counts = np.ones(len(df['flake_Dmax']))
        
    area = counts

    # normalize scatter sizes (so sizes are comparable across plots)
    scaler = MinMaxScaler(feature_range=(20, 200))  # size range
    s_scaled = scaler.fit_transform(area.reshape(-1, 1)).ravel()

    if variable == 'gan3d_mass':
        bounds = np.arange(0, 37.5, 2.5)
    elif variable == 'fallspeed':
        bounds = np.arange(0, 1000, 25)
    norm = BoundaryNorm(bounds, ncolors=256, extend='max')

    fig, ax = plt.subplots(1, 2, figsize=(12,5), )

    
    # (a) log-log fit
    sc = ax[0].scatter(D*1e3, m, label="Data", c=counts,cmap='Grays', norm=norm, s=s_scaled, alpha=0.7, edgecolor='k')
    ax[0].plot(10**logD*1e3, 10**log_lin_fit, 'b--', label=f'linear fit: b ={b_lin:.2f}')
    ax[0].plot(10**logD*1e3, 10**log_quad_fit, 'r-', label=f'quadratic fit')

    ax[0].set_xlabel("D [mm]")
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(f"{var_name}-dimension fits")
    # first legend: fits
    legend1 = ax[0].legend(loc="upper left")
    ax[0].grid(True, ls=':')
    # ax[0].set_yscale('log')
    
    if 'count_flake_Dmax' in df.columns:
        # add colorbar for counts
        cbar=fig.colorbar(sc, ax=ax[0], boundaries=bounds, ticks=bounds[::2], spacing="uniform", format="%.0f")
        cbar.set_label("Counts")
        # cbar.ax.set_yticklabels((bounds[::2]))
    
    
    # (b) local exponent vs D
    idx = np.argsort(b_local)
    b_sorted = b_local.iloc[idx]
    D_valid = D[b_sorted.index]

    ax[1].plot(D_valid*1e3, b_sorted, 'ro-', label='quadratic slope b(D)')
    ax[1].hlines(b_lin, D.min()*1e3, D.max()*1e3, colors='b', linestyle='--', label=f'linear fit: b ={b_lin:.2f}')
    ax[1].set_xlabel("D [mm]")
    ax[1].set_ylabel("Exponent b")
    ax[1].set_title("Diameter-dependent exponent")
    ax[1].legend()
    ax[1].grid(True, ls=':')

    fig.suptitle(f"{title}", fontweight ="bold",y=0.96)  
    plt.tight_layout(rect=[0,0,1,1])

def find_dimension_fits_sigma(df, ds):
    ### find the sigma-dimension relationship from reflectivity
    # PSD
    n_per_m_all = ds['n_per_m'].values 
    # n_per_m_all [particles m^-3 per m]  -> n_per_mm_all [particles m^-3 per mm]
    n_per_mm_all = n_per_m_all / 1000.
    # observed reflectivity
    Z_obs_all_mm = df['Ze_linear_sum_400'].values
    Z_obs_all_m = Z_obs_all_mm*1e-18

    dD_m = ds['dD'].values #difference between bin_edges
    dD_mm = dD_m*1000.     # convert to mm

    D_m = ds['bin'].values
    D_mm = D_m*1000.    #convert to mm

    

    # Replace Nans in PSD with 0 (assume no particles in that bin)
    n_per_mm_all = np.nan_to_num(n_per_mm_all, nan=0.0)

    # use exsisting counts for weighting
    # counts = df['count_flake_Dmax']
    # # normalize weights
    # # weights = np.nansum(counts, axis=1) # one weight per PSD bin
    # # weights = weights/np.nanmax(weights)

    # Mask invalid time steps (any NaN in PSD or Z)
    valid = (~np.isnan(Z_obs_all_mm)) & (~np.isnan(n_per_mm_all).all(axis=1)) #& (counts > 0)
    n_per_mm_all = n_per_mm_all[valid, :]
    Z_obs_all_mm = Z_obs_all_mm[valid]
    # counts = counts[valid]

    # # normalize counts to prevent huge differences
    # weights = counts/counts.max()

    

    # Find inital guess for a0 and b0
    logD_list, logsigma_list = [], []
    for i in range(len(Z_obs_all_mm)):
        n = n_per_mm_all[i, :] # since n can be zero the nansum in sigma_pseudo will return -inf, hence not only nan but alos inf have to be masked
        
        Z = Z_obs_all_mm[i]

        # approximate sigma 
        sigma_pseudo = Z * (n *dD_mm) / np.nansum(n*dD_mm)
        logD_list.append(np.log10(D_mm))   # logD = np.log10(D)
        logsigma_list.append(np.log10(sigma_pseudo))

    logD = np.concatenate(logD_list)
    logs = np.concatenate(logsigma_list)

    idx = np.argsort(logD)
    logD_sorted = logD[idx]

    mask = (~np.isnan(logD)) & (~np.isnan(logs)) & (~np.isinf(logs))
    logD, logs = logD[mask], logs[mask]

    # linear fit (deg=1)
    lin_coeff = np.polyfit(logD, logs, 1) # slope, intercept
    b0_lin, a0_lin = lin_coeff
    # print(10**a0_lin, b0_lin)

    # quadratic fit (deg=2)
    quad_coeff = np.polyfit(logD, logs, 2) # c2, c1, c0
    c2, c1, c0 = quad_coeff
    b_local = local_exponent(D_mm, quad_coeff)
    def residuals(params):
        a, b = params
        # compute PSD contribution X_t(b) for each time sample Z_pred(t) = a * sum(D^b N dD)
        X = np.nansum((D_mm**b)[None,:] * n_per_mm_all * dD_mm[None,:] , axis=1) # weighted PSD contribution
        Z_pred = a * X
        # return np.sqrt(w) * (Z_pred - Z_obs_all_mm)  # return vector of residuals
        return (Z_pred - Z_obs_all_mm)
        

    p0 = (max(10**a0_lin, 1e-12), max(b0_lin,1e-6)) # initial guess for the parameters the solver will try to fit
    # p0 = (a0_lin, b0_lin)
    # starting guess for cross-section scaling: a_sigma = 1e-6, 
    #starting guess for the exponent: b_sigma=2

    # least_squares minimizes min sum(Z_pred - Z_obs)^2
    # res = least_squares(residuals, x0=np.array(p0), args=(weights,), bounds=([0,0], [np.inf, np.inf]))
    res = least_squares(residuals, x0=np.array(p0),bounds=([0,0], [np.inf, np.inf]))
    # guarangees both a and b are positive # bounds=([lower_a, lower_b], [upper_a, upper_b]) defines the feasible region
    # # a_sigma >= 0 (backscatter cross section can't be negative)
    # # b_sigma [-10,10] prevent numerical instability
    a_lin, b_lin = res.x
    # print(f'sigma = {10**a_lin:.2f} D$^{b_lin:.2f}$')

    # 2. Compute fitted curve and local exponent
    # Fitted curve in log-log space
    log_lin_fit = np.log10(a_lin * (D_mm**b_lin))
    log_quad_fit = np.polyval(quad_coeff, np.log10(D_mm))

    return logD_sorted, log_lin_fit, log_quad_fit, b_local, b_lin, a_lin


def plt_residuals_dimension_fits(ds, df, log_quad_fit_z, b_local_z, a_lin_z, b_lin_z, var_name, title=" "):
    a_lin = a_lin_z
    b_lin = b_lin_z
    b_local = b_local_z
    D_m = ds['bin'].values
    D_mm = D_m*1000.    #convert to mm
    # PSD
    n_per_m_all = ds['n_per_m'].values 
    # n_per_m_all [particles m^-3 per m]  -> n_per_mm_all [particles m^-3 per mm]
    n_per_mm_all = n_per_m_all / 1000.
    dD_m = ds['dD'].values #difference between bin_edges
    dD_mm = dD_m*1000.     # convert to mm
    Z_obs_all_mm = df['Ze_linear_sum_400'].values
    Z_obs_all_m = Z_obs_all_mm*1e-18

    # linear fit predicted Z (a*D^b)
    Z_pred_lin = a_lin * np.nansum((D_mm**b_lin)[None,:] * n_per_mm_all * dD_mm[None,:], axis=1)

    # quadratic fit
    sigma_quad = 10**log_quad_fit_z    # σ(D) per bin
    Z_pred_quad = np.nansum(sigma_quad[None, :] * n_per_mm_all *dD_mm[None, :], axis=1)

    mask = (~np.isnan(Z_obs_all_mm)) & (~np.isnan(Z_pred_lin)) & (~np.isnan(Z_pred_quad))
    Z_obs_all_mm, Z_pred_lin, Z_pred_quad = Z_obs_all_mm[mask], Z_pred_lin[mask], Z_pred_quad[mask]
    r2_lin = r2_score(Z_obs_all_mm, Z_pred_lin)
    r2_quad = r2_score(Z_obs_all_mm, Z_pred_quad)

    # Scatter plot 
    fig, ax = plt.subplots(1, 3, figsize=(18,5), )
    ax[0].scatter(Z_obs_all_mm, Z_pred_lin, alpha=0.5, color='b', label=f'linear fit: b ={b_lin:.2f}, R²={r2_lin:.3f}')
    ax[0].scatter(Z_obs_all_mm, Z_pred_quad, alpha=0.5, color='r', label=f'quadratic fit: R²={r2_quad:.3f}')
    # 1:1 line
    x = np.linspace(0, 10e4, 100)
    y = x
    ax[0].plot(x, y, linestyle='--', color='darkgrey')  # 1:1

    ax[0].set_xlabel("Observed Z [mm$^6$ m$^{-3}$]")
    ax[0].set_ylabel("Predicted Z [mm$^6$ m$^{-3}$]")
    ax[0].set_title(f"{var_name}-Fit performance")

    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].grid(True, ls=':')

    # (b) Residuals (Observed - Predicted)
    res_lin = Z_obs_all_mm - Z_pred_lin
    res_quad = Z_obs_all_mm - Z_pred_quad

    ax[1].scatter(Z_obs_all_mm, res_lin, alpha=0.5, color='b', label=f'linear residuals')
    ax[1].scatter(Z_obs_all_mm, res_quad, alpha=0.5, color='r', label=f'quadratic residudals') # residuals mostly negative at large Z, that means the model systematically underestimates for large reflectivity.
    ax[1].axhline(0, color="k", ls="--")
    ax[1].set_xlabel("Observed Z [mm$^6$ m$^{-3}$]")
    ax[1].set_ylabel("Residual (Obs - Pred)")
    ax[1].set_title("Residuals vs Observed Z")
    ax[1].legend()
    ax[1].grid(True, ls=":")

    # (b) local exponent vs D
    # sort D for smooth plotting
    idx = np.argsort(D_mm)
    D_sorted = D_mm[idx]    # bin center
    b_sorted = b_local[idx]

    ax[2].plot(D_sorted, b_sorted, 'ro-', label='quadratic slope b(D)')
    ax[2].hlines(b_lin, D_sorted.min(), D_sorted.max(), colors='b', linestyle='--', label=f'linear fit: b ={b_lin:.2f}')
    ax[2].set_xlabel("D [mm]")
    ax[2].set_ylabel("Exponent b")
    ax[2].set_title("Diameter-dependent exponent")
    ax[2].legend()
    ax[2].grid(True, ls=':')

    for axs in ax[:2]:
        axs.set_xscale('log')
        # axs.set_yscale('log')
    
    fig.suptitle(f"{title}", fontweight ="bold",y=0.96)  
    plt.tight_layout(rect=[0,0,1,1])
    # plt.tight_layout()




def apply_timeoffset_to_df(df, df_sum_offset, df_mean_offset, min_rate=0.1, a=56, b=1.2):


    
    # Drop rows where there is no fallspeed registered
    # df = df[df['fallspeed'].notna()]
    seconds_to_subtract = 400 / df['fallspeed']

    # Convert seconds into timedeltas
    timedeltas = pd.to_timedelta(seconds_to_subtract, unit='sec')
    # convert to minutes as a float, round, and then convert back to timedelta64[m]
    rounded_td = (timedeltas / np.timedelta64(1,'m')).round().astype('timedelta64[m]')

    # subtract from index
    df['shifted_time_end'] = df.index - rounded_td

    # if len(df_sum_offset) == 0:
    #     df_sum_offset = ds[['Matrosov_a_400', 'Bracci_a_400', 'Nakai_a_400', 'Nakai_g_400', 'Pu_g_400', f'S_{a}_{b}_400',
    #                 'Matrosov_a_800', 'Bracci_a_800', 'Nakai_a_800', 'Nakai_g_800', 'Pu_g_800', f'S_{a}_{b}_800',
    #                 'Matrosov_a_1000', 'Bracci_a_1000', 'Nakai_a_1000', 'Nakai_g_1000', 'Pu_g_1000', f'S_{a}_{b}_1000',
    #                 'Ze_linear_400', 'Ze_linear_800', 'Ze_linear_1000']].sel(time=slice('2023-02-15', '2023-03-14')).to_dataframe()
    df_sum_offset = df_sum_offset.copy()
    df_sum_offset.rename(columns={'Ze_linear_400': 'Ze_linear_sum_400', 'Ze_linear_800': 'Ze_linear_sum_800', 'Ze_linear_1000': 'Ze_linear_sum_1000'}, inplace=True)
    # df_sum_offset['S_sim_all'] = ((df_sum_offset['Ze_linear_sum_400'] / 273.33) ** (1/0.31)) 
    # df_sum_offset['S_sim_aggregate'] = ((df_sum_offset['Ze_linear_sum_400'] / 5006.19) ** (1/1.26))  
    # df_sum_offset['S_sim_graupel'] = ((df_sum_offset['Ze_linear_sum_400'] / 316.87) ** (1/0.39)) 
    # df_sum_offset['S_sim_mixed'] = ((df_sum_offset['Ze_linear_sum_400'] / 492.65) ** (1/0.57)) 
    # if len(df_mean_offset) == 0:
    #     df_mean_offset = ds[['Ze_400m', 'Ze_linear_400', 'Ze_linear_800', 'Ze_linear_1000', 'W_400m', 'W_800m', 'W_1000m']].sel(time=slice('2023-02-15', '2023-03-14')).to_dataframe()

    df_mean_offset = df_mean_offset.copy()
    # for alt in [400, 800, 1000]:
    alt = 400
    # Limit the fallspeed from MRR to "realistic values"
    mask = df_mean_offset[f'W_{alt}m'].abs() > 7.
    df_mean_offset.loc[mask, f'W_{alt}m'] = np.nan
            # 

    # df_mean_offset.rename(columns={'Ze_linear_400': 'Ze_linear_mean_400', 'Ze_linear_800': 'Ze_linear_mean_800', 'Ze_linear_1000': 'Ze_linear_mean_1000'}, inplace=True)

    
   
    sums = []
    means= []
    zs_count = []
    for i in range(len(df)):
        # seg = df_sum_offset.loc[df['shifted_time_end'][i] + pd.Timedelta(seconds=1) : df.index[i]]
        # seg_m = df_mean_offset.loc[df['shifted_time_end'][i] + pd.Timedelta(seconds=1) : df.index[i]]
        delta_h_i = df['delta_h'].iloc[i]
        shifted_end = df['shifted_time_end'].iloc[i]

        step = delta_h_i * 60 - 1
        start = shifted_end - pd.Timedelta(minutes=step)
        seg = df_sum_offset.loc[start : shifted_end]
        seg_m = df_mean_offset.loc[start : shifted_end]
        
    
        # print(df.index[i], seg)
        sums.append(seg.sum(numeric_only=True, min_count=1))
        means.append(seg_m.mean(numeric_only=True, skipna=True))

        # zs_cc = seg[['Ze_linear_sum_400', 'Ze_linear_sum_800', 'Ze_linear_sum_1000']].count()
        # zs_cc = zs_cc.rename({"Ze_linear_sum_400":"count_Ze_lin_400", "Ze_linear_sum_800":"count_Ze_lin_800", "Ze_linear_sum_1000":"count_Ze_lin_1000"})
        # zs_count.append(zs_cc)


    df_offset = pd.concat([df[['shifted_time_end', 'Geonor', 'air_temperature', 'wind_direction', 'wind_speed', 'snowflake_class_name_60', 'snowflake_class_name_70', 'snowflake_class_name_80', 'fallspeed', 'flake_Dmax', 'gan3d_mass' ,'delta_h', 'rate_mm_h', 'count_fallspeed', 'count_flake_Dmax', 'count_gan3d_mass', 'count_precip']],
                        #    pd.DataFrame(zs_count).set_index(df.index[:]),
                                    pd.DataFrame(sums).set_index(df.index[:]), 
                                    pd.DataFrame(means).set_index(df.index[:]), ], axis=1)# df['shifted_time_end'][1:].reset_index(drop=True)], axis=1)

    # limit Geonor precipitation to be above 0.1 mm/h
    # df_offset = df_offset[df_offset["rate_mm_h"] >= 0.1]
    df_offset[df_offset["rate_mm_h"] < min_rate] = np.nan
    df_offset['start_time'] = df_offset.index - pd.to_timedelta(df_offset['delta_h'] * 3600, unit='sec')


    # for alt in [400, 800, 1000]:
        # # Compute Z from linear sums
        # # Compute safely: return NaN for invalid values
        # df_offset[f'Z_from_linear_sum_{alt}'] = df_offset[f'Ze_linear_sum_{alt}'].apply(lambda y: 10 * np.log10(y) if y > 0 else np.nan)
        # df_offset[f'Z_from_linear_mean_{alt}'] = df_offset[f'Ze_linear_mean_{alt}'].apply(lambda y: 10 * np.log10(y) if y > 0 else np.nan)
    alt = "400"
    for name in ["Matrosov_aggregate", "Bracci_aggregate", "Nakai_aggregate", "Nakai_graupel", "Pu_graupel", "Schoger",f"S_{a}_{b}"]:
            df_offset[f"{name}_{alt}_mm_h"] = df_offset[f"{name}_{alt}m"] / df_offset["delta_h"]

        

        # # Calculate Matrosov based on Ze_sum
        # df_offset[f'Matrosov_from_ze_sum_{alt}'] = ((df_offset[f'Ze_linear_sum_{alt}'] / 56) ** (1/1.2)) /(60*df_offset["delta_h"])  # convert rate to mm

    # for class_name in ['all', 'aggregate', 'graupel', 'mixed']:
    #     df_offset[f'S_sim_{class_name}_mm_h'] = df_offset[f'S_sim_{class_name}'] / df_offset["delta_h"]
    return df_offset

# https://stackoverflow.com/questions/75731161/how-to-calculate-a-linear-regression-with-a-predefined-slope
def sse_intercept(loga, b, logS, logZ, w):
    Z_pred = loga + b*logS
    residuals = (Z_pred - logZ)
    # weighted_sse = np.sum(w * residuals**2)
    weighted_sse = np.sum(residuals**2)
    return weighted_sse # weighted residuals


def edges_from_centers(centers):
    centers = np.asarray(centers)
    d = np.diff(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = centers[:-1] + d / 2
    edges[0] = centers[0] - d[0] / 2
    edges[-1] = centers[-1] + d[-1] / 2
    return edges

def find_bin(val, edges):
    return np.searchsorted(edges, val) -1 

def draw_lit_box(ax, box, a_edges, b_edges, lw=2.5):
    ia = find_bin(box["a"], a_edges)
    ib = find_bin(box["b"], b_edges)

    x0 = b_edges[ib]
    y0 = a_edges[ia]
    width  = b_edges[ib + 1] - b_edges[ib]
    height = a_edges[ia + 1] - a_edges[ia]

    ax.add_patch(Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        edgecolor=box["color"],
        linewidth=lw,
        zorder=11
    ))

def draw_lit_box_index(ax, box, a_vals, b_vals):
    a = box["a"]
    b = box["b"]

    # find nearest indices
    iy = np.argmin(np.abs(a_vals - a))
    ix = np.argmin(np.abs(b_vals - b))

    ax.add_patch(Rectangle(
        (ix, iy), 1, 1,
        fill=False,
        edgecolor=box["color"],
        linewidth=2.5,
        zorder=11
    ))

LITERATURE_BOXES = {
    "A": dict(a=56, b=1.20, 
              color="#C51B7D", #"lime",    
              label="Matrosov (2007)"),
    "B": dict(a=134,   b=1.25, 
              color="#F768A1",#"yellow",  
              label="Bracci et al. (2022)"),
    "C": dict(a=252,   b=2.03, 
              color="#FDB863",#"magenta", 
              label="Pu et al. (2020)"),
    "D": dict(a=77.61, b=1.22, 
              color="#00BFC4",#"cyan",    
              label="Schoger et al. (2021)"),
}

LIT_PLACEMENT = {
    # Physical categories
    ("aggregate", "graupel", "mixed", "campaign"): {
        "aggregate": ["A", "B"],
        "graupel":   ["C"],
        "campaign":  ["D"],
    },
    ("campaign"): {
        "campaign":  ["D"],
    },
    # Wind-direction categories
    ("North", "East", "South", "West"): {
        "North": ["A", "B", "C", "D"],
        "East":  ["A", "B", "C", "D"],
        "South": ["A", "B", "C", "D"],
        "West":  ["A", "B", "C", "D"],
    },
}

def get_lit_placement(category):
    key = tuple(category)
    if key not in LIT_PLACEMENT:
        raise ValueError(f"No literature placement defined for category={category}")
    return LIT_PLACEMENT[key]


def plt_chessboard(df, counts, cbar_label, category, vmin = 0,vmax=1):
    cmap = cm.managua  # define the colormap
    # extract all colors from the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    custom_cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 14)

    custom_colors = {  
                     category[0]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(1),  
                     category[1]:LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(4),   
                     category[2]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(2),
                     category[3]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(5),
}
    
    
    wanted_yvals = [2, 4, 6, 8, 10, 50, 100, 150, 200, 250, 300]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,12), sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_2d(axs)
    for (i, ax), class_key in zip(enumerate(axs.flatten()), category):
        a_vals = df[class_key].index.values
        b_vals = df[class_key].columns.values

        # a_edges = edges_from_centers(a_vals)
        # b_edges = edges_from_centers(b_vals)
        x_edges = np.arange(len(b_vals) +1)
        y_edges = np.arange(len(a_vals) +1)

        # A, B = np.meshgrid(df[class_key].columns, df[class_key].index)
        # A, B = np.meshgrid(np.arange(len(b_vals)), 
        #                    np.arange(len(a_vals)))
        c = ax.pcolormesh(x_edges, y_edges, df[class_key].values, vmin=vmin, vmax=vmax, 
                      cmap=custom_cmap,shading="auto", edgecolors=(0, 0, 0, 0.2), linewidth=0.005, alpha=0.9)
        
        # hightlight the cell with minimum RMSE/MAE
        # index of min
        # iy, ix = df[class_key].stack().idxmin()
        iy, ix= np.unravel_index(np.nanargmin(abs(df[class_key].values)), df[class_key].shape)
        # rectangle edges
        ax.add_patch(Rectangle((ix, iy), 1, 1, # index space
                     fill=False, edgecolor="black", linewidth=3.5, zorder=9))
        
        ax.add_patch(Rectangle((ix,iy), 1, 1,
                     fill=False, edgecolor="white", linewidth=2.5, zorder=10))

        # # Highlight literature boxes
        a_to_idx = {a:i for i,a in enumerate(a_vals)}
        b_to_idx = {b:i for i,b in enumerate(b_vals)}
        placement = get_lit_placement(category)
        for study_key in placement.get(class_key, []):
            box = LITERATURE_BOXES[study_key]
            draw_lit_box_index(ax, box, a_vals, b_vals)

  


        # Ticks & labels
        ax.set_title(f"{class_key} (counts={counts[class_key]})".title(), fontweight='bold', color=custom_colors[class_key])
        ax.text(0.05, 1.025, fig_label[i], fontweight="bold", horizontalalignment="center", verticalalignment='center', transform = ax.transAxes,color="black")

        x_idx = np.arange(2, len(b_vals), 5)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(b_vals[x_idx])
        # a_vals is non-uniform, select by value, map to index
        y_idx = [i for i, a in enumerate(a_vals) if a in wanted_yvals]
        ax.set_yticks(y_idx)
        ax.set_yticklabels(a_vals[y_idx])

        split_idx = np.where(a_vals == 10)[0][0]
        ax.axhline(split_idx, color='white', lw=2)

    for i in range(2):
        axs[i, 0].set_ylabel(r"$a$", fontsize=20, )#color='white')
        axs[1, i].set_xlabel(r"$b$", fontsize=20, )#color='white')
    
    lit_handles = [
    Patch(facecolor="none",
          edgecolor=box["color"],
          linewidth=2.5,
          label=box["label"])
    for box in LITERATURE_BOXES.values()
]


    fig.legend(
    handles=lit_handles,
    loc='upper left',
    bbox_to_anchor=(0.88, 0.28),  # tweak this
    frameon=True,
    title=r"Literature ($a$, $b$)"
)

    

    plt.tight_layout(rect=[0, 0, 0.9, 1])   # leave room for cbars
    cax = fig.add_axes([0.9, 0.3 , 0.015, 0.45]) # left, bottom, width, height
    cb = fig.colorbar(c, cax=cax, orientation='vertical', label=cbar_label,spacing="proportional", extend="both")

def extract_best_ab(df_metrics):
    """
    Extract best (a,b) per [metric_key][class_key] from stacked DataFrames.
    Returns: dict[metric_key][class_key] = (a, b)
    """
    result = {}
    for metric_key, class_dict in df_metrics.items():
        result[metric_key] = {}
        for thresh in class_dict.keys():
            result[metric_key][thresh] = {}
            for class_key, df in class_dict[thresh].items():
                if df.empty:
                    continue
                df = df.abs()
                stacked = df.stack()

                if stacked.empty or stacked.isna().all():
                    continue
                result[metric_key][thresh][class_key] = stacked.idxmin()

    return result


def build_pp(cfg, source_dict, delta_h, thresh, time_index):

    aligned = []
    for cls, (a, b) in cfg.items():
        key = f"S_{a}_{b}_400_mm_h"
        s = source_dict[thresh][cls][key]

        # KEEP NaNs here
        s = s.reindex(time_index)

        aligned.append(s)

    # combine WITHOUT forcing zeros
    pp = pd.concat(aligned, axis=1).sum(axis=1, min_count=1)

    return pp * delta_h.reindex(time_index)

def plt_cum(ax, series, time_index, label, color, ls, lw, alpha=None):
    s = series.reindex(time_index)
    s = s.fillna(0)   # correct for rates / increments
    cum = s.cumsum()
    cum = cum - cum.iloc[0]   # normalize start to 0
    # ax.plot(cum, label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
    ax.step(cum.index, cum, where='post', label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha)

def plot_literature_lines(ax, class_key, category_set, x_line):
    """
    ax           : matplotlib axis
    class_key    : current subplot category (e.g. 'aggregate', 'North')
    category_set : tuple used in LIT_PLACEMENT keys
    x_line       : x grid (logspace)
    """

    if category_set not in LIT_PLACEMENT:
        return

    placement = LIT_PLACEMENT[category_set]

    if class_key not in placement:
        return

    for lit_id in placement[class_key]:
        lit = LITERATURE_BOXES[lit_id]

        a = lit["a"]
        b = lit["b"]
        color = lit["color"]
        label = lit["label"]

        y = a * x_line**b

        ax.plot(
            x_line, y,
            color=color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=label
        )

def plt_scatter_SWEobs_SWEret_4x4(rmse_ab, df_rmse, df, category):
    custom_colors = {  
                     category[0]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(1),  
                     category[1]:LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(4),   
                     category[2]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(2),
                     category[3]: LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)(5),
}

    fig, axs = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True, constrained_layout=True)

    for (i, ax), class_key in zip(enumerate(axs.flatten()), category):
        a, b = rmse_ab[class_key]

        rmse_val = df_rmse[class_key].loc[a, b]
        x_obs = df[class_key]['rate_mm_h']
        y_ret = df[class_key][f"S_{a}_{b}_400_mm_h"]
        sc = ax.scatter(x_obs, y_ret, color=custom_colors[class_key], alpha=0.7, edgecolors=custom_colors[class_key])


        
        # plot 1:1 line
        x_line = np.linspace(0.01, 40, 100)
        ax.plot(x_line, x_line, linestyle='--', color='darkgrey', lw=1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, ls=':', which='both')


        # fixed limits (important for visiual stability)
        ax.set_xlim(0.009, 40)
        ax.set_ylim(0.009, 40)
        
        ax.text(0.05, 0.95, fig_label[i], fontweight="bold", horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)

        title = ax.set_title("", fontweight='bold', color=custom_colors[class_key])
        class_key = class_key.replace("_", " ").title()
        title.set_text(f"{class_key} — a = {a},  b = {b}")

        # dynamic annotation box
        text_box = ax.text(0.12, 0.98, "", 
                transform=ax.transAxes, va='top', ha='left', fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

        
        # update annotation
        text_box.set_text(
                f"a = {a}\n"
                f"b = {b:.2f}\n"
                f"RMSE = {rmse_val:.4f}"
            )

    for i in range(2):
        axs[i, 0].set_ylabel("SWE$_{retrieved}$ [mm h$^{-1}$]", )#fontsize=20)
        axs[1, i].set_xlabel("SWE$_{obs}$ [mm h$^{-1}$]",)# fontsize=20)