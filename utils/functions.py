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
                                x='time', y='range', 
                                vmin=-20, levels=np.arange(-20,30,0.5), 
                                cmap=plt.get_cmap('pymeteo_radar'), 
                                extend='max', add_colorbar=False)
        cbaxes = fig.add_axes([0.92, 0.8, 0.025, 0.14])
        cbar = plt.colorbar(cf1, cax=cbaxes, shrink=0.5, orientation='vertical')
        cbar.ax.set_ylabel('Ze ({})'.format(ds['Ze'].attrs['units']))

        # second plot Doppler velocity
        cf2 = ds['W'].plot.contourf(ax = axs[1], 
                                        x='time', y='range',
                                        vmin=-3, vmax=3, levels=np.arange(-3,3.5,0.5), 
                                        extend='both', cmap=cm.vik, add_colorbar=False,)
        cbaxes = fig.add_axes([0.92, 0.625, 0.025, 0.14])
        cbar = plt.colorbar(cf2, cax=cbaxes, shrink=0.5, orientation='vertical')
        cbar.ax.set_ylabel('W ({})'.format(ds['W'].attrs['units']))

        # third plot precipitation accumulation hourly
        axs[2].bar(ds['sum(precipitation_amount PT1H)'].time.values, ds['sum(precipitation_amount PT1H)'].values, width=0.025, align='center')
        axs[2].set_ylabel('Precipitation ({})'.format(ds['sum(precipitation_amount PT1H)'].attrs['units']))
        axs[2].set_ylim([0, 4])

        # forth plot 2-m temperature
        ds['air_temperature'].plot(ax = axs[3], x='time')
        axs[3].set_ylabel('2-m temperature ({})'.format(ds['air_temperature'].attrs['units']))
        axs[3].hlines(y= 2., xmin=day_start, xmax=xmax, color = 'grey')
        axs[3].set_ylim([-11.5, 12.5])

        # fifth plot 10-m wind
        color = 'tab:blue'
        axs[4].scatter(x=ds['wind_from_direction'].time.values, y =ds['wind_from_direction'].values, color=color,)
        axs[4].set_ylabel('Wind direction ({})'.format(ds['wind_from_direction'].attrs['units']),color=color, )
        axs[4].tick_params(axis='y', labelcolor=color)
        axs[4].set_ylim(0, 360)

        ax2 = axs[4].twinx()
        color = 'tab:green'
        ds['wind_speed'].plot(ax=ax2, x='time', color=color)
        ax2.set_ylabel('Wind speed ({})'.format(ds['wind_speed'].attrs['units']), color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0,20)

        # Define the date format
        date_form = DateFormatter("%H:00")
        for ax, tt in zip(axs,title):
                ax.set( xlabel = "",
                        title =tt)
                ax.xaxis.grid()
                ax.set_xlim([day_start, _xmax])
                ax.tick_params(axis='x', labelrotation=0, )
                plt.setp(ax.get_xticklabels(), horizontalalignment="center")
                ax.xaxis.set_major_formatter(date_form)
        for ax in axs[0:2]:
                        ax.set(ylabel = "Height [m]", yticklabels=ds['height'].isel(time=0).values[::5])
        
        axs[-1].set(xlabel = "Time [UTC]",)
            
        fig.suptitle("{} ({})".format(site.split('.')[0], ds.time.dt.date[0].values), fontweight="bold");
        fig.subplots_adjust(top=0.94)

