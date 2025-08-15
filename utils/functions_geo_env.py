from imports_geo_env import (plt, ccrs, np, gaussian_kde, xr, find_peaks,)

fig_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)',
             '(aa)', '(bb)', '(cc)', '(dd)', '(ee)', '(ff)', '(gg)', '(hh)', '(ii)', '(jj)', '(kk)', '(ll)', '(mm)', '(nn)', '(oo)', '(pp)', '(qq)', '(rr)', '(ss)', '(tt)', '(uu)', '(vv)', '(ww)', '(xx)', '(yy)', '(zz)']


def add_base_map(ax, ax_extent, x_coords, y_coords, data, land_gdf, lon0, lat0, cmap='terrain'):
    ax.set_extent(ax_extent, crs=ccrs.PlateCarree())
    # Use pcolormesh to plot the cropped raster
    mesh = ax.pcolormesh(x_coords, y_coords, data, transform=ccrs.PlateCarree(),
                      cmap=cmap, shading='auto',alpha=0.8)
    
    # Add land and ocean geometries
    for geom in land_gdf.geometry:
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1, alpha=0.5, label='land')
        
    # Add a marker for the center point
    plt.scatter(lon0, lat0, color='red', transform=ccrs.PlateCarree(), marker='^', s=100)
    
    return mesh


def add_ray_directions(ax, lon0, lat0, results, switch_indices):
    """Add rays and angle annotations"""
    for angle, label, met_angle in results:
        theta = np.deg2rad(angle)
        lon1 = lon0 + 0.25 * np.cos(theta)
        lat1 = lat0 + 0.25 * np.sin(theta)
        color = 'brown' if label == 'land' else 'blue' if label == 'ocean' else 'gray'
                
        # # Plot the ray line
        # ax.plot([lon0, lon1], [lat0, lat1], color = color, transform=ccrs.PlateCarree(), linewidth=2, alpha=0.5, label = label if label not in ax.get_legend_handles_labels()[1] else "")
        
        # Only annoate selecte angles
        if angle in [results[i][0] for i in switch_indices]:
            # Plot the ray line
            ax.plot([lon0, lon1], [lat0, lat1], color = color, transform=ccrs.PlateCarree(), linewidth=2, alpha=0.5, label = label if label not in ax.get_legend_handles_labels()[1] else "")
            ax.text(lon1, lat1, f"{met_angle}°", fontsize=14, fontweight='bold', ha='center', va='center', color='black', transform=ccrs.PlateCarree())
    ax.legend(loc='upper left', fontsize=10, frameon=True, title="Ray Classification", title_fontsize='13', markerscale=1.5)


        
def add_gridlines(ax):
    """Grid and styling"""
    ax.background_patch = None
    gl = ax.gridlines(draw_labels=True,linewidth=0.5, color='gray', alpha=0.5, linestyle='--') # 
    #  Disable labels on the top and right
    gl.top_labels   = False
    gl.right_labels = False
    # Set tick parameters to hide ticks on the top and right
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
    
def plt_histogram_kde(ax, data, bins, color, xlabel, xlim=None, ylim=None, kde=True,):# shade=None):
    """
    Plot histogram + KDE on a given axis.

    Parameters:
    - ax: matplotlib axis
    - data: data array
    - bins: histogram bins
    - color: base color
    - xlabel: label for x-axis
    - xlim, ylim: optional x/y limits
    - kde: whether to plot KDE
    - shade: list of (xmin, xmax, color, label) to shade regions
    """
    # Histogram
    kde_results = None

    ax.hist(data, bins=bins, density=True, alpha=0.3, label=xlabel, histtype='bar', align='mid', rwidth=0.9, color=color, edgecolor='black')
    # KDE
    if kde:
        kde_curve = gaussian_kde(data)
        x_grid = np.linspace(bins[0], bins[-1], 200)
        ax.plot(x_grid, kde_curve(x_grid), color=color)
        kde_results = {'x_grid': x_grid, 'kde': kde_curve(x_grid)}
        
        
    # Axes labels and limits
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hist. Density")
    ax.grid(alpha=0.5)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        
    return kde_results

def plt_variable_group(axs, data_dict, var_names, bins_dict, colors, labels, xlims, ylims, vlines=None, kde=False, shade_regions=None, legend_title=None, var_units=None, show_titles=True):
    kde_results = None
    suffixes = ["_land", "_ocean", "_cold", "_warm", "_cao", "_wai", "_q25", "_q75", "_undef", "_filled", "_other"]

    if kde:
        kde_results = {}
        for label in data_dict.keys():
            kde_results[label] = {}
            
                    
    for j, (label, data) in enumerate(data_dict.items()):
        for i, var in enumerate(data.keys()):
            values = data[var]
            if len(values) == 0:
                continue
            if 'wind_direction' in var:
                axs[i].axvline(x=vlines[0], color='grey', alpha=0.5)
                axs[i].axvline(x=vlines[1], color='grey', alpha=0.5)

            for suffix in suffixes:
                if var.endswith(suffix):
                    var = var.removesuffix(suffix)

            kde_result = plt_histogram_kde(axs[i], values, bins_dict[var], colors[label], labels.get(label, label), xlim=xlims[var], ylim=ylims[var], kde=kde)
            if kde:
                kde_results[label][var] = {}
                kde_results[label][var] = kde_result
            axs[i].set_xlabel('')
                
        for i, var in enumerate(var_names):
            if (i==0 and shade_regions):
                ymin = ylims[var][1]*0.92
                ymax = ylims[var][1]*0.99
                for start, end, color_shade, wdlabel in shade_regions:
                    axs[i].fill_between([start, end], ymin, ymax, color=color_shade, alpha=0.3)
                    axs[i].text((start+end)/2, (ymin + ymax)/2, wdlabel, ha='center', va='center', fontsize=9, color=color_shade, fontweight='bold')
            if show_titles:
                unit = f" ({var_units[var]})" if var_units and var in var_units else ""
                axs[i].set_title(f"{var.replace('_', ' ').title()}{unit}")
                

            
            
    if legend_title:
        axs[1].legend(title=legend_title, title_fontsize='13') 
        
    # axs[-1].set_xlabel(var_names[-1].replace('_', ' ').title())
    return kde_results

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
                    
def convert_dict_to_xarray(kde_vars, kde_results):
    
    # Fill containers
    for label, var_dict in kde_results.items():
        for var, kde_dict in var_dict.items():
            # print(label, var)
            if var in kde_vars:  # make sure it’s one of the 3 we're tracking
                kde_vars[var][label] = xr.DataArray(
                    data=kde_dict['kde'],
                    dims=['x'],
                    coords={'x': kde_dict['x_grid']},
                    name=label
                )
    return kde_vars

# Custom function to convert meteorological degrees (0° N, clockwise)
# to matplotlib degrees (0° E, counterclockwise)
def met_to_math_deg(theta_met):
    return (90 - theta_met) % 360

def plt_kde_WD_WD_2TM_land_ocean_sf_class_cold_warm(data_set, instrument, xlims, ylims, bins_dict, shade_regions, var_units, vlines):
    # Create containers for each variable
    kde_vars = {
        'wind_direction': {},
        'wind_speed': {},
        'air_temperature': {}
    }
    
    fig, axsm = plt.subplots(4, 3, figsize=(15, 12), )
    axs = axsm.flatten()

    kde_results = plt_variable_group(axs[0:3],{instrument: extract_clean_data(data_set, ['wind_direction', 'wind_speed', 'air_temperature']),}, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={instrument:'green'}, labels={instrument: instrument}, xlims=xlims, ylims=ylims, kde=True, shade_regions=shade_regions, var_units=var_units, legend_title='Instrument instrument', vlines=vlines, show_titles=True)

    kde_vars = convert_dict_to_xarray(kde_vars, kde_results)
    
    # ---- Plot now land/ocean points KDE
    # Data dicts by condition
    data_dict = {
        'land': extract_clean_data(data_set, ['wind_direction_land', 'wind_speed_land', 'air_temperature_land']),
        'ocean': extract_clean_data(data_set, ['wind_direction_ocean', 'wind_speed_ocean', 'air_temperature_ocean'])
    }
    kde_results = plt_variable_group(axs[3:6], data_dict, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={'land':'sienna', 'ocean':'dodgerblue'}, labels={'land':'Land', 'ocean':'Ocean'}, xlims=xlims, ylims=ylims, kde=True,  legend_title ='Wind direction condition', vlines=vlines, var_units=var_units, show_titles=False)
    kde_vars = convert_dict_to_xarray(kde_vars, kde_results)
    
    # ---- Plot snowfall class aggregate/graupel KDE
    data_dict = {
        'aggregate': extract_clean_data(data_set, ['wind_direction', 'wind_speed', 'air_temperature'], mask=(data_set['most_common_class']=='aggregate')),
        'graupel':   extract_clean_data(data_set, ['wind_direction', 'wind_speed', 'air_temperature'], mask=(data_set['most_common_class']=='graupel')),
        'pristine':  extract_clean_data(data_set, ['wind_direction', 'wind_speed', 'air_temperature'], mask=(data_set['most_common_class'] =='pristine'))

        }
    kde_results = plt_variable_group(axs[6:9], data_dict, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={'aggregate':'darkviolet', 'graupel':'darkorange', 'pristine':'teal'}, labels={'aggregate':'Aggregates', 'graupel':'Graupel', 'pristine':'Pristine'}, xlims=xlims, ylims=ylims, kde=True,  legend_title ='Snowflake Class Name', vlines=vlines, var_units=var_units, show_titles=False)
    kde_vars = convert_dict_to_xarray(kde_vars, kde_results)
    
    # ---- Plot temperature class cold/warm KDE
    # Data dicts by condition
    data_dict = {
        'cold': extract_clean_data(data_set, ['wind_direction_cold', 'wind_speed_cold', 'air_temperature_cold']),
        'warm': extract_clean_data(data_set, ['wind_direction_warm', 'wind_speed_warm', 'air_temperature_warm'])
        }
    kde_results = plt_variable_group(axs[9:12], data_dict, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={'cold':'mediumblue', 'warm':'crimson'}, labels={'cold':'cold', 'warm':'warm'}, xlims=xlims, ylims=ylims, kde=True,  legend_title ='Temperature condition', vlines=vlines, var_units=var_units, show_titles=False)
    kde_vars  = convert_dict_to_xarray(kde_vars, kde_results)
    
    for i, ax in enumerate(axs):
        ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    
    plt.tight_layout()

    return kde_vars


def find_primary_secondary_max(wind_direction_ds, wind_speed_ds, air_temperature_ds):
    for ds_name, dataset in zip(['Wind direction', 'Wind speed', 'Air temperature'], [wind_direction_ds, wind_speed_ds, air_temperature_ds]):
        for var_name, da in dataset.data_vars.items():
            kde_vals = dataset[var_name]
            x_vals = dataset['x']
            
            peaks, _ = find_peaks(kde_vals)
        
            # Sort peaks by height
            peak_heights = kde_vals[peaks]
            sorted_idx = np.argsort(peak_heights)[::-1]
            
            
            # Primary max
            max1_idx = peaks[sorted_idx[0]]
            print(f"Primary max {ds_name} {var_name}: {x_vals[max1_idx]:.2f} (density={kde_vals[max1_idx]:.3f})")
            # Secondary max (if exists)
            if len(sorted_idx) > 1:
                max2_idx = peaks[sorted_idx[1]]
                print(f"Secondary max {ds_name} {var_name}: {x_vals[max2_idx]:.2f} (density={kde_vals[max2_idx]:.3f})")
            else:
                print(f"No secondary max for {ds_name} {var_name}")
                
def plt_kde_WD_WD_2TM_cold_warm_Tq25T_q75(data_set, instrument, xlims, ylims, bins_dict, shade_regions, var_units, vlines):
    # Create containers for each variable
    kde_vars = {
        'wind_direction': {},
        'wind_speed': {},
        'air_temperature': {}
    }
    
    fig, axsm = plt.subplots(3, 3, figsize=(15, 9), )
    axs = axsm.flatten()

    kde_results = plt_variable_group(axs[0:3],{instrument: extract_clean_data(data_set, ['wind_direction', 'wind_speed', 'air_temperature']),}, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={instrument:'green'}, labels={instrument: instrument}, xlims=xlims, ylims=ylims, kde=True, shade_regions=shade_regions, var_units=var_units, legend_title='Instrument Location', vlines=vlines, show_titles=True)

    kde_vars = convert_dict_to_xarray(kde_vars, kde_results)


    # Data dicts by condition
    data_dict = {
        'cold': extract_clean_data(data_set, ['wind_direction_cold', 'wind_speed_cold', 'air_temperature_cold']),
        'warm': extract_clean_data(data_set, ['wind_direction_warm', 'wind_speed_warm', 'air_temperature_warm'])
    }
    kde_results = plt_variable_group(axs[3:6], data_dict, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={'cold':'mediumblue', 'warm':'crimson'}, labels={'cold':'cold', 'warm':'warm'}, xlims=xlims, ylims=ylims, kde=True,  legend_title ='Temperature condition', vlines=vlines, var_units=var_units, show_titles=False)
    kde_vars  = convert_dict_to_xarray(kde_vars, kde_results)

    # Data dicts by condition
    data_dict = {
        'T_q25': extract_clean_data(data_set, ['wind_direction_q25', 'wind_speed_q25', 'air_temperature_q25']),
        'T_q75': extract_clean_data(data_set, ['wind_direction_q75', 'wind_speed_q75', 'air_temperature_q75'])
    }
    kde_results = plt_variable_group(axs[6:9], data_dict, ['wind_direction', 'wind_speed', 'air_temperature'], bins_dict, colors={'T_q25':'mediumblue', 'T_q75':'crimson'}, labels={'T_q25':'T_q25', 'T_q75':'T_q75'}, xlims=xlims, ylims=ylims, kde=True,  legend_title ='Temperature condition', vlines=vlines, var_units=var_units, show_titles=False)
    kde_vars  = convert_dict_to_xarray(kde_vars, kde_results)

    for i, ax in enumerate(axs):
        ax.text(0.05, 0.95, fig_label[i], fontweight='bold', horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
        
    plt.tight_layout()
    return kde_vars

