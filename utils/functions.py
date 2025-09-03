# from imports import *
# from imports import requests, BeautifulSoup
# import imports
from imports import (xr, np,datetime, datetime,cm,plt,DateFormatter, pd,da, BeautifulSoup,BoundaryNorm, LinearSegmentedColormap,requests,date2num,)
fig_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)',
             '(aa)', '(bb)', '(cc)', '(dd)', '(ee)', '(ff)', '(gg)', '(hh)', '(ii)', '(jj)', '(kk)', '(ll)', '(mm)', '(nn)', '(oo)', '(pp)', '(qq)', '(rr)', '(ss)', '(tt)', '(uu)', '(vv)', '(ww)', '(xx)', '(yy)', '(zz)']

height_bins = np.arange(0, 3000, 100) # Height bins [m] bins of 100 m for height 
ze_bins = np.arange(-10,31,1)# Reflectivity bins [dBZ] 1 dBZ for reflectivity
W_bins = np.arange(-3,3,0.075) # bin widths of 0.075 m s−1 for Doppler velocity
sW_bins = np.arange(0,1.2, 0.05) # bin widths of 0.05 m s−1 for Doppler spectral width



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
    # Get the nearest indices
    nearest_indices = da.indexes['time'].get_indexer(variable.time.values, method='nearest')
    
    # Assign values to the appropriate time index
    da.data[nearest_indices] = variable.data
    
    return da

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
    # Remove Nans and flatten the data
    var1 = ds[var_name1].values.flatten()
    var2 = ds[var_name2].values.flatten()
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
def ze_to_snowfall_rate(Ze_dbz):
    # Ze to linear units
    Ze_lin = 10 ** (Ze_dbz /10)
    # Apply Z-S relationship
    S = (Ze_lin / 56) ** (1/1.2)                # Matrosov (2007) - aggregate snowflakes
    S_bracci = (Ze_lin / 134) ** (1/1.25)       # Bracci et al. (2021) - aggregate
    S_nakai_a = (Ze_lin / 107.4) ** (1/1.67)    # Nakai et al. (2021) - heavily rimed aggregate
    S_nakai_g = (Ze_lin / 227.4) ** (1/1.67)    # Nakai et al. (2021) - graupel
    S_pu = (Ze_lin / 252) ** (1/2.03)           # Pu et al. (2020) - graupel


    # convert rate per minute to mm/h
    S = (S/60)#.resample(time='1H').sum()
    S_bracci = (S_bracci/60)#.resample(time='1H').sum()
    S_nakai_a = (S_nakai_a/60)#.resample(time='1H').sum()
    S_nakai_g = (S_nakai_g/60)#.resample(time='1H').sum()
    S_pu = (S_pu/60)#.resample(time='1H').sum()
    
    df1 = S.isel(range=3).drop('range').rename('Matrosov_aggregate').to_dataframe()
    
    df2 = S_bracci.isel(range=3).drop('range').rename('Bracci_aggregate').to_dataframe()
    
    df3 = S_nakai_a.isel(range=3).drop('range').rename('Nakai_aggregate').to_dataframe()
    
    df4 = S_nakai_g.isel(range=3).drop('range').rename('Nakai_graupel').to_dataframe()
    
    df5 = S_pu.isel(range=3).drop('range').rename('Pu_graupel').to_dataframe()
    
    
    combined_df = pd.concat([df1, df2, df3, df4, df5], axis=1) # Use axis=1 to combine as new columns
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
    ('pristine', 3)
    # ('planar_crystal', 5),
    # ('small_particle', 6)
]
    markers = ['*', '^',  'h','o',  's', 'd']
    
    for y_pos, (name, level) in enumerate(particle_classes, 1) :
        data = ds[name].dropna('time')
        ax.plot(date2num(data['time']), 
                np.full(len(data['time']), level),
                marker=markers[y_pos - 1], linestyle='None',
                color=custom_colors[name], markersize=4, label=name)
    # ax.legend(loc='best', )#fontsize=8)
    ax.set_yticks(range(1, 4))
    # ax.set_yticklabels("") 
    ax.set_yticklabels([name for name, _ in particle_classes])
    ax.grid(True, ls=':')


custom_colors = {  
                     'aggregate': cm.lipariS(2),  
                     'graupel':cm.lipariS(4),   
                     'mixed': cm.lipariS(3),
                     'pristine':cm.lipariS(0),
                
                
                }

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
    fig, axsm = plt.subplots(3, 1, figsize=(8.27, 7.014), sharex=True, )
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

    format_range_axis(axs[0], ds)

    # snowflake class
    plot_snowflake_classes(axs[1], ds)

    
    # Precipitation accumulation hourly
    precip = ds['precipitation'].resample(time='1H').sum('time', skipna=True, keep_attrs=True)
    axs[2].bar(precip.time.values, precip.values, width=np.timedelta64(40, 'm'), align='center', color='tab:blue', label='Valid Precipitation', alpha=0.5)
    # Highlight missing data periods with shaded background
    missing_times = precip['time'][np.where(precip==0.)].values
    for t in missing_times:
            axs[2].axvspan(t- np.timedelta64(20, 'm'), t + np.timedelta64(20, 'm'), color='gray', alpha=0.2)
    axs[2].set_ylabel(f"Precipitation [{ds['precipitation'].attrs['units']}]")
    axs[2].set_ylim([0, 4.1])
    axs[2].legend()

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
#     # Doppler velocity
#     bounds  = np.arange(-6,7,1)
#     norm    = BoundaryNorm(boundaries=bounds, ncolors=256)
#     cmap    = cm.vik
    
# #     dopp    = axs[2].pcolormesh(time_vals, ds['range'], ds['W'].transpose('range', 'time'), cmap=cmap, norm=norm, shading='auto')
#     plot_pcolormesh_with_cbar(ax=axs[2], fig=fig, data=ds['W'].transpose('range', 'time'), x=time_vals, y=ds['range'], cmap=cmap, norm=norm, cbar_bounds=np.arange(-6, 8, 2), cbar_label=f"Doppler Velocity [{ds['W'].attrs['units']}]", cbar_pos=[cbar_left, cbar_top -2 * cbar_step, cbar_width, cbar_height])
    
#     format_range_axis(axs[2], ds)

    
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
    for ax, title in zip(axs, titles):
        ax.set(xlabel = "", title=title)
        ax.xaxis.grid(ls=':')
        ax.tick_params(axis='x', labelrotation=0)
        plt.setp(ax.get_xticklabels(), horizontalalignment="center")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xlim([day_start, xmax])

    axs[-1].set(xlabel="Time [UTC]")
    fig.suptitle(f"{site.split('.')[0]} ({ds.time.dt.date[0].values})", fontweight="bold")
    # fig.subplots_adjust(top=0.94)
    fig.subplots_adjust(top=0.92)

