from imports import *

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

def plt_Ze_W_p_T_U(ds, day_start, xmax, site):

        title = ['MRR reflectivity', 'MRR mean Doppler Velocity', 'Precipitation accumulation', '2-m Air temperature', '10-m Wind']
        fig, axsm = plt.subplots(5, 1, figsize=[10,12], sharex=True, )
        axs = axsm.flatten()

        # first plot reflectivity
        cf1 = ds['Ze'].plot.contourf(ax = axs[0], 
                                x='time', 
                                vmin=-20, levels=np.arange(-20,30,0.5), 
                                cmap=plt.get_cmap('pymeteo_radar'), 
                                extend='max', add_colorbar=False)
        cbaxes = fig.add_axes([0.92, 0.8, 0.025, 0.14])
        cbar = plt.colorbar(cf1, cax=cbaxes, shrink=0.5, orientation='vertical')
        cbar.ax.set_ylabel('Ze ({})'.format(ds['Ze'].attrs['units']))

        # second plot Doppler velocity
        cf2 = ds['W'].plot.contourf(ax = axs[1], 
                                        x='time',
                                        vmin=-3, vmax=3, levels=np.arange(-3,3.5,0.5), 
                                        extend='both', cmap=cm.vik, add_colorbar=False,)
        cbaxes = fig.add_axes([0.92, 0.625, 0.025, 0.14])
        cbar = plt.colorbar(cf2, cax=cbaxes, shrink=0.5, orientation='vertical')
        cbar.ax.set_ylabel('W ({})'.format(ds['W'].attrs['units']))

        # third plot precipitation accumulation hourly
        axs[2].bar(ds['precipitation'].time.values, ds['precipitation'].values, width=0.025, align='center')
        axs[2].set_ylabel('Precipitation ({})'.format(ds['precipitation'].attrs['units']))
        axs[2].set_ylim([0, 4])

        # forth plot 2-m temperature
        if len(ds['air_temperature'].dropna(dim='time')) != 0:
                ds['air_temperature'].dropna(dim='time').plot(ax = axs[3], x='time')
                axs[3].hlines(y= 2., xmin=day_start, xmax=xmax, color = 'grey')
        
        axs[3].set_ylabel('2-m temperature ({})'.format(ds['air_temperature'].attrs['units']))
        axs[3].set_ylim([-11.5, 12.5])

        # fifth plot 10-m wind
        color = 'tab:blue'
        if len(ds['wind_direction'].dropna(dim='time')) !=0:
                axs[4].scatter(x=ds['wind_direction'].dropna(dim='time').time.values, y =ds['wind_direction'].dropna(dim='time').values, color=color,)
        axs[4].set_ylabel('Wind direction ({})'.format(ds['wind_direction'].attrs['units']),color=color, )
        axs[4].tick_params(axis='y', labelcolor=color)
        axs[4].set_ylim(0, 360)

        ax2 = axs[4].twinx()
        color = 'tab:green'
        if len(ds['wind_speed'].dropna(dim='time')) !=0:
                ds['wind_speed'].dropna(dim='time').plot(ax=ax2, x='time', color=color)
        ax2.set_ylabel('Wind speed ({})'.format(ds['wind_speed'].attrs['units']), color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0,20)

        # Define the date format
        date_form = DateFormatter("%H:00")
        # if len(ds['air_temperature'].dropna(dim='time')) != 0
        for ax, tt in zip(axs,title):
                ax.set( xlabel = "",
                                title =tt)
                ax.xaxis.grid()
                ax.tick_params(axis='x', labelrotation=0, )
                plt.setp(ax.get_xticklabels(), horizontalalignment="center")
                ax.xaxis.set_major_formatter(date_form)
                if len(ds['air_temperature'].dropna(dim='time')) != 0 or len(ds['wind_direction'].dropna(dim='time')) !=0 or len(ds['wind_speed'].dropna(dim='time')) !=0:
                        ax.set_xlim([day_start, xmax])
        for ax in axs[0:2]:
                        ax.set(ylabel = "Height [m]", yticklabels=ds['height'].isel(time=0).values[::5])
        
        axs[-1].set(xlabel = "Time [UTC]",)
            
        fig.suptitle("{} ({})".format(site.split('.')[0], ds.time.dt.date[0].values), fontweight="bold");
        fig.subplots_adjust(top=0.94)



def arrays_same_time_res(variable, da):
    for i in range(variable.time.size):
        idx = np.where((variable.time.isel(time=i) == da.time) ==True)[0][0]
        da[idx] = variable.isel(time=i)

    return(da)


def MRR_bin_averaged_dataset(ds, dsm):
    ## Average MRR variables in range bins
    # Radar equivalent reflectivity ($Ze$) in dBz is transformed to MRR radar reflectivity ($Z$) 
    # in $mm^6 m^{-3}$ by the following relationship:
    # # $$Ze = 10log_{10} (\frac{Z}{mm^6 m^{-3}})$$
    # # $$Z = 10^{Ze/10} mm^6 m^{-3}$$
    # # Once we know $Z$ it is possible to average over range sizes. The assumption is that the 4th 
    # range represents the bin between 300 and 400m and so on. Reflectivity, Doppler velocity, and height will 
    # be averaged in the bins (4,5), (6,7), (8,9), ... (26,27), (28,29), (30,31)
    
    _Ze_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])), fill_value=np.nan), coords ={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
    _W_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])), fill_value=np.nan), coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
    _height_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])),fill_value=np.nan),coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
    _sWidth_mean = xr.DataArray(data=da.full(shape = (len(ds.time),len(ds.range[1:-1][::2])),fill_value=np.nan),coords={'time':ds.time.values, 'range':ds.range[1:-1][::2]})
    for i, k in zip(ds.range[1:-1][::2].values, range(len(ds.range[1:-1][::2]))):
                _Z = 10**(ds['Ze'].sel(range=slice(i, (i+1)))/10)
                _Z_mean = _Z.mean('range', skipna=True, keep_attrs=True)
                _Ze_mean[:, k] = 10*np.log10(_Z_mean)

                _W_mean[:,k] = ds['W'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)
                _height_mean[:,k] = ds['height'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)
                _sWidth_mean[:,k] = ds['spectralWidth'].sel(range=slice(i, (i+1))).mean('range', skipna=True, keep_attrs=True)


    dsm['Ze'] = _Ze_mean
    dsm['W']  = _W_mean
    dsm['height'] = _height_mean
    dsm['spectralWidth'] = _sWidth_mean
        
    dsm['Ze'].attrs = ds['Ze'].attrs
    dsm['W'].attrs = ds['W'].attrs
    dsm['height'].attrs = ds['height'].attrs
    dsm['spectralWidth'].attrs = ds['spectralWidth'].attrs

    return dsm