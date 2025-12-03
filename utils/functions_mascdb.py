# from imports import *
# from imports import requests, BeautifulSoup
# import imports
from imports_mascdb import (cm, np, ListedColormap,BoundaryNorm, plt, sns, pd, Patch, OrderedDict, mdates)
import pathlib


fig_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)',
             '(aa)', '(bb)', '(cc)', '(dd)', '(ee)', '(ff)', '(gg)', '(hh)', '(ii)', '(jj)', '(kk)', '(ll)', '(mm)', '(nn)', '(oo)', '(pp)', '(qq)', '(rr)', '(ss)', '(tt)', '(uu)', '(vv)', '(ww)', '(xx)', '(yy)', '(zz)']

custom_colors = {  
                     'aggregate': cm.lipariS(2),  
                     'graupel':cm.lipariS(4),   
                     'mixed': cm.lipariS(3),
                     'pristine':cm.lipariS(0),
                
                
                }

def wind_sector(deg):
    if (deg >= 337.5) or (deg < 22.5):
        return 'N'
    elif 22.5 <= deg < 67.5:
        return 'NE'
    elif 67.5 <= deg < 112.5:
        return 'E'
    elif 112.5 <= deg < 157.5:
        return 'SE'
    elif 157.5 <= deg < 202.5:
        return 'S'
    elif 202.5 <= deg < 247.5:
        return 'SW'
    elif 247.5 <= deg < 292.5:
        return 'W'
    else:
        return 'NW'

def get_wind_sectors(df, direction_labels, col_in_name='env_DD_test', col_out_name='wind_direction_sector'):
    num_bins = len(direction_labels)
    bin_width_deg = 360/num_bins
    half_bin = bin_width_deg /2
    bin_edges = np.linspace(-half_bin, 360 - half_bin, num_bins + 1)
    
    df[col_out_name] = pd.cut(df[col_in_name], bins=bin_edges, labels = direction_labels, include_lowest=True)
    
    return(df)


def plt_snowflake_heatmap_by_sector(df, mode='percent', class_ratio=None, title="Snowflake Class Heatmap",
                                        cmap_base=cm.davos_r, n_bins=10, data_max = 100, figsize=(10,4), cbar_label=None, annotate=True, reorder_sectors=True):
                                        
    """
    Plots a heatmap of snowflake class frequency across wind direction sectors.

    Parameters:
    - df: DataFrame with ['wind_direction_sector', 'snowflake_class_name']
    - mode: "percent", "absolute", or "ratio"
    - class_ratio: tuple of two class names if mode="ratio" (e.g. ("graupel", "aggregate"))
    - title: Title of the plot
    - cmap_base: Base matplotlib colormap (e.g. cm.davos_r)
    - n_bins: Number of bins for the color scale
    - figsize: Size of the figure
    - cbar_label: Label for the colorbar (autodetected if None)
    - annotate: Whether to annotate values in the heatmap
    - reorder_sectors: If True, orders sectors N, NE, E, ...
    """
    #Group counts Compute Snowflake Frequencies per ( Wind Sector)
    counts = df.groupby(['wind_direction_sector', 'snowflake_class_name']).size().reset_index(name='count')
    # Pivot to a contingency table: rows=(temp, wind), columns=snowflake_class
    pivot = counts.pivot_table(index=['wind_direction_sector'], columns='snowflake_class_name', values='count', fill_value=0)
    
    # Optional reordering
    if reorder_sectors:
        ordered_sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        pivot = pivot.reindex(index=ordered_sectors, fill_value=0)
        
    # Mode selection
    if mode == "percent":
        pivot_plot = pivot.div(pivot.sum(axis=1), axis=0).T *100
        default_label = "% frequency"
        format = ".2f"
    elif mode == "absolute":
        pivot_plot = pivot.T
        default_label = "Snowflake Count"
        format = ".0f"
    elif mode == "ratio":
        if class_ratio is None or len(class_ratio) != 2:
            raise ValueError("Provide class_ratio=(numerator_class, denominator_class)")
        # Create graupel:aggregate ratio per wind sector
        numerator, denominator = class_ratio
        ratio = pivot[numerator] / pivot[denominator]
        ratio.replace([np.inf, -np.inf], np.nan, inplace=True)  # handle zero division
        pivot_plot = ratio.to_frame(name=f"{numerator}/{denominator}").T
        default_label = f"Ratio"
        format = ".2f"
    else:
        raise ValueError("Mode must be one of 'percent', 'absolute', or 'ratio'")
            
    # Create discrete colormap
    if mode == "percent":
        bounds = np.linspace(0, data_max, n_bins+1)
    elif mode == "absolute":
        # data_max = np.nanmax(pivot_plot.values)
        # data_max = 2500
        bounds = np.linspace(0, data_max, n_bins + 1)
    if mode == "ratio":
        # data_max = np.nanpercentile(pivot_plot.values, 95)
        # data_max = 6.5
        bounds = np.linspace(0, data_max, n_bins + 1)
    

    
    cmaplist = [cmap_base(i/(n_bins -1)) for i in range(n_bins)]
    custom_cmap = ListedColormap(cmaplist)
    
    norm = BoundaryNorm(bounds, ncolors=custom_cmap.N)
    
    # Plot 
    plt.figure(figsize=figsize)
    # ax = sns.heatmap(pivot_plot, annot=annotate, fmt=format, cmap=custom_cmap, norm=norm, cbar_kws={'label': cbar_label or default_label})
    if annotate and mode == "percent":
        # Format: "XX.X%\n(n=YY)"
        annot_data = pivot_plot.round(1).astype(str) + "%\n(" + pivot.T.astype(int).astype(str) + ")"
        fmt = ""
    elif annotate:
        annot_data = True  # default behavior
        fmt = format
    else:
        annot_data = False

    ax = sns.heatmap(pivot_plot, annot=annot_data, fmt=fmt, cmap=custom_cmap, norm=norm,
                    cbar_kws={'label': cbar_label or default_label})

    # Labeling
    ax.set_ylabel("Snowflake Class" if mode != "ratio" else "")
    ax.set_xlabel("Wind Direction Sector")
    ax.set_title(title)

    plt.tight_layout()
    
def plt_dominant_snowflake_class_by_sector(pivot_table, reorder_sectors = True, class_colors=None, cmap=cm.davos, figsize = (10,4), annotate=True, title="Dominant Snowflake Class by Wind Sector"): 
    """
    Plots a bar chart showing the dominant snowflake class per wind sector.

    Parameters:
    - pivot_table: DataFrame with wind sectors as index and snowflake classes as columns (counts).
    - ordered_sectors: Optional list of wind directions in desired compass order.
    - class_colors: Optional dict mapping class name to color (e.g., {'graupel': ..., 'aggregate': ...}).
    - cmap: Colormap used to derive colors (only if class_colors not provided).
    - figsize: Tuple for figure size.
    - annotate: If True, adds class name and percent above bars.
    - title: Title of the plot.
    """
    
    
    
    # Optional reordering
    if reorder_sectors:
        ordered_sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    
    # Find the dominant class per sector
    dominant_class = pivot_table.idxmax(axis=1)   # dominant class name (e.g., 'graupel', 'aggregate', etc.)
    dominant_count = pivot_table.max(axis=1)
    total_count = pivot_table.sum(axis=1)
    dominant_fraction = pivot_table.max(axis=1) / pivot_table.sum(axis=1)   # how dominant that class is (from 0 to 1)
    
    # Dataframe
    dominant_df = pd.DataFrame({'wind_sector':dominant_class.index, 'dominant_class': dominant_class.values, 'dominant_count': dominant_count.values, 'dominant_fraction': dominant_fraction.values})
    
    # Reorder wind sectors
    dominant_df['wind_sector'] = pd.Categorical(dominant_df['wind_sector'], categories=ordered_sectors, ordered=True)

    dominant_df = dominant_df.sort_values('wind_sector')
    
    # Define colors if none provided
    if class_colors is None:
        unique_classes = dominant_df['dominant_class'].unique()
        cmap_vals = [i / (len(unique_classes) -1 or 1) for i in range(len(unique_classes))]
        class_colors = {cls: cmap(val) for cls, val in zip(unique_classes, cmap_vals)}
        
    # Map colors
    bar_colors = dominant_df['dominant_class'].map(class_colors)
    
    # Bar plot shows fraction of dominance and class per wind sector
    plt.figure(figsize=figsize)
    bars = plt.bar(dominant_df['wind_sector'], dominant_df['dominant_fraction'], color=bar_colors)
    
    # Add legend manually
    for cls, color in class_colors.items():
        plt.bar(0, 0, color=color, label=cls)
    plt.legend(title='Dominant Class', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Add text annotation
    if annotate:
        for i, row in dominant_df.iterrows():
            label = f"{row['dominant_class']} ({int(row['dominant_count'])})"
            plt.text(row['wind_sector'], row['dominant_fraction'] + 0.1, label, ha='center', va='bottom', fontsize=9)
            
    # Lables and style 
    plt.ylabel("Fraction")
    plt.xlabel("Wind Direction Sector")
    plt.ylim(0, 1.15)
    plt.grid(True, ls=':')
    plt.title(title)
    plt.tight_layout()


def plt_dominant_snowflake_class_heatmap(pivot_table,reorder_sectors = True, class_colors=None, figsize = (10,2), annotate=True, title="Dominant Snowflake Class by Wind Sector (Heatmap)"):
    
    """
    Plots a heatmap showing the dominant snowflake class per wind sector.

    Parameters:
    - pivot_table: DataFrame of counts with index=wind_direction_sector, columns=snowflake_class_name.
    - reorder_sectors: Optional list of wind directions in desired compass order.
    - class_colors: Dict mapping snowflake class names to colors.
    - title: Plot title.
    - figsize: Size of the figure.
    - annotate: Whether to annotate each bin with class name and dominance %.
    """
    
    # Optional reordering
    if reorder_sectors:
        ordered_sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
    # Find the dominant class per sector
    dominant_class = pivot_table.idxmax(axis=1)   # dominant class name (e.g., 'graupel', 'aggregate', etc.)
    dominant_count = pivot_table.max(axis=1)
    dominant_fraction = pivot_table.max(axis=1) / pivot_table.sum(axis=1)   # how dominant that class is (from 0 to 1)
    
    dominant_class = dominant_class[ordered_sectors]
    dominant_fraction = dominant_fraction[ordered_sectors]
    dominant_count = dominant_count[ordered_sectors]
    
    # Define colors if none provided
    cmap=cm.davos
    if class_colors is None:
        unique_classes = dominant_class.unique()
        class_colors = {cls: cmap(i / (len(unique_classes) - 1 or 1)) for i, cls in enumerate(unique_classes)}

    
    # Map class names to integers
    class_to_int = {cls: i for i, cls in enumerate(class_colors)}
    int_row = [class_to_int[cls] for cls in dominant_class]
    int_data = pd.DataFrame([int_row], columns=ordered_sectors)


    # Custom colormap and ensure fixed order
    sorted_classes = sorted(class_to_int.items(), key=lambda x: x[1])
    cmap = ListedColormap([class_colors[cls] for cls, _ in sorted_classes])     #cmap[0] corresponds to class_to_int['graupel'] == 0
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(int_data, cmap=cmap, cbar=False, linewidths=0.5, linecolor='white', xticklabels=True, yticklabels=False, vmin=0, vmax=len(class_colors)-1)

    # Annotate class names and % dominance
    if annotate: 
        for j, sector in enumerate(ordered_sectors):
            dom_cls = dominant_class[sector]
            frac = dominant_fraction[sector] * 100
            count = dominant_count[sector]
            ax.text(j + 0.5, 0.5, f"{frac:.0f}% ({count})", ha='center', va='center', color='white', fontsize=9)

    # Formatting
    ax.set_xticklabels(ordered_sectors)
    ax.set_ylabel('')
    ax.set_title(title)
    
    # Legend
    legend_patches = [Patch(color=color, label=cls) for cls, color in class_colors.items()]
    plt.legend(handles=legend_patches, title="Class", bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    
    
def plt_stacked_snowflake_windrose(df, direction_column='env_DD_test', class_column='snowflake_class_name', class_color=None, direction_labels=None, title="Wind Rose of Crystal Types", figsize=(8,7), mode="count" # Options: "count", "total", "per_direction"
                                   ):
    """
    Plots a stacked polar bar chart (windrose) of snowflake crystal type counts by wind direction.

    Parameters:
    - df: DataFrame with at least wind direction and snowflake class columns.
    - direction_column: Name of the wind direction column (degrees).
    - class_column: Name of the snowflake class column.
    - class_colors: Dict mapping class names to colors.
    - direction_labels: Optional list of 16 compass labels (e.g., ['N', 'NNE', ...]).
    - title: Plot title.
    - figsize: Size of the figure.
    - mode: One of "count" (absolute), "total" (normalized globally), or "per_direction" (normalize within sector).
    """
    
    if direction_labels is None:
        direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',]
        
    
    # Bin edges in degrees, 22.5° width: center N on 0°
    num_bins = len(direction_labels)
    bin_width_deg = 360/num_bins
    half_bin = bin_width_deg /2
    # bin_edges = np.arange(-half_bin, 360+half_bin+0.1, bin_width_deg)
    bin_edges = np.linspace(-half_bin, 360 - half_bin, num_bins + 1)
    # bin_edges = np.arange(-11.25, 371.25, 22.5)
    # Bin wind direction into compass labels
    df = df.copy()
    # df['compass'] = pd.cut(df[direction_column], bins=bin_edges, labels= direction_labels, include_lowest=True)
    
    # count occurences by class and compass
    count_df = df.groupby(['compass', df[class_column]]).size().unstack(fill_value=0).reindex(direction_labels, fill_value=0)
    
    if mode == "total":
        count_df = count_df / count_df.values.sum() * 100 # normalize to percentages
    elif mode == "per_direction":
        count_df = count_df.div(count_df.sum(axis=1), axis=0).fillna(0) *100
    
    # convert to individual class series
    angles = np.deg2rad(np.arange(0, 360, bin_width_deg))
    width = np.deg2rad(bin_width_deg)
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # stacked bars
    bottom = np.zeros_like(angles)
    for snow_class in count_df:
        bar_height = count_df[snow_class].values
        color = class_color.get(snow_class, 'gray') if class_color else None
        ax.bar(angles, bar_height, width=width, bottom=bottom, color = color, edgecolor=None, label=snow_class,)
        bottom += bar_height
        
    # Compass labels
    # ax.set_thetagrids(np.degrees(angles), labels=direction_labels)
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    # ax.set_thetagrids(np.degrees(angles), labels=direction_labels)
    ax.set_thetagrids(np.degrees(angles))
    ax.set_xticklabels(direction_labels)
    ax.grid(True, ls=':')
    ax.set_title(f"{title} ({mode.replace('_', ' ').title()})", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    
    # if mode != "count":
    # if mode == "total":
    #     ax.set_ylim(0, 35)
    #     ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
    #     ax.set_yticklabels(['5%', '10%', '15%', '20%', '25%', '30%', '35%'])
    if mode == "per_direction":
        ax.set_ylim(0,100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'])
    plt.tight_layout()

def plt_timeseries_temp_snowflake_class(df_env, class_color=custom_colors, title="Snowflake Class Temperature"):
    fig, ax = plt.subplots(figsize=(15,5))
    # for (class_name, da), marker, color in zip(snowflake_class.items(), markers,colors):
    for class_name, color in class_color.items():
        df_env.loc[df_env['snowflake_class_name'] == class_name].plot(y='env_T_test', marker='+', label=class_name, linestyle='None', color=color, markersize=10, ax=ax, alpha=0.8)


    ax.hlines(y=4., xmin=df_env.index[0], xmax=df_env.index[-1], colors='gray', linestyles=":")
    ax.hlines(y=2., xmin=df_env.index[0], xmax=df_env.index[-1], colors='gray', linestyles=":")

    ax.set_ylabel('env T [$^\circ$C]')


        
    # Set the x-axis labels daily using DateFormatter and DayLocator
    ax.legend()
    ax.grid(True, ls=':')
    # ax.set_xlim([start_date, end_date])  # Set xlim to specific date range
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Set ticks for every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format labels as dates (Year-Month-Day)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")  # Rotate the labels for better readability
    ax.set_ylim([-6, 3])
    ax.set_xlabel('Datetime')
    ax.set_title(f"{title}", pad=20)

    plt.tight_layout()

def plt_timeseries_wind_snowflake_class(df_env, class_color=custom_colors, title="Snowflake Class Wind Direction"):
    fig, ax = plt.subplots(figsize=(15,5))
    # for (class_name, da), marker, color in zip(snowflake_class.items(), markers,colors):
    for class_name, color in class_color.items():
        df_env.loc[df_env['snowflake_class_name'] == class_name].plot(y='env_DD_test', marker='+', label=class_name, linestyle='None', color=color, markersize=10, ax=ax, alpha=0.8)



    ax.set_ylabel('env T [$^\circ$C]')



    # Set the x-axis labels daily using DateFormatter and DayLocator
    ax.legend()
    ax.grid(True, ls=':')
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Set ticks for every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format labels as dates (Year-Month-Day)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")  # Rotate the labels for better readability
    ax.set_yticks(np.arange(0,365,45))
    ax.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
    ax.set_xlabel('Datetime')
    ax.set_title(f"{title}", pad=20)

    plt.tight_layout()

def summarize_env_by_dominant_class(df, frequency=None, dominant_frac_threshold=None):
    
    
    # if frequency is None:
    #     # Just group by original index without resampling
    #     sf_class_count = df.groupby([df.index, 'snowflake_class_name']).size().unstack(fill_value=0)
    # else:
    sf_class_count = df.groupby([pd.Grouper(freq=frequency), 'snowflake_class_name']).size().unstack(fill_value=0)


    
    # Normalize each row to get fractions
    df_total_sf = sf_class_count.sum(axis=1)
    df_frac = sf_class_count.div(df_total_sf, axis=0)

    # find the max class to get fraction
    dominant_class = df_frac.idxmax(axis=1)
    dominant_frac = df_frac.max(axis=1)

    # Apply thresholds only if provided
    if dominant_frac_threshold is None:
        counts = dominant_class.rename('most_common_class')
    else:
        counts = dominant_class.where(dominant_frac >= dominant_frac_threshold, 'mixed').rename('most_common_class')
    
    
    # In every time bin assign the mean temperature, wind speed, direction, and riming degree where the snowflake class name is dominant
    results = []
    for i in range(len(counts) -1):
        # Define time window
        t_start = counts.index[i]
        t_end = counts.index[i+1]

        # Subset original data to current 1-min window
        _minute = df[(df.index >= t_start) & (df.index < t_end)]

        # Get dominant snowflake class label for current time bin
        snowflake_class = counts.loc[t_start]

        # Filter _minute by this snowflake class name, but use entire minute if the snowflake class name is mixed
        if snowflake_class == 'mixed':
            _minute = _minute.copy()
        else:
            _minute =  _minute[_minute['snowflake_class_name'] == snowflake_class]

        if _minute.empty:
            continue # skip if no data in this time range

        # compute mean of numeric values
        mean_values = _minute.mean(axis=0, skipna=True)

        # add the class label as a series
        mean_values['snowflake_class_name'] = snowflake_class

        # store as row in final result list with t_start as index
        results.append(pd.DataFrame([mean_values], index=[t_start]))
        
    # Combine all into a DataFrame
    df_env_threshold = pd.concat(results)
    df_env_threshold = pd.merge(df_env_threshold, sf_class_count, left_index=True, right_index=True)
    
    return df_env_threshold

def test_dominance_threshold(df, frequency, dominant_frac_threshold, fig_dir):
    
    
    # df_nan = df.dropna()
    # # create a Boolean mask for the rows to remove
    # mask = df['snowflake_class_name'] == ''
    # # # select all rows except the ones that contain ''
    # df_nan = df_nan[~mask]
    df_nan = df.copy()
    
    if frequency is None:
        df_env = df_nan.copy()
    else:
        df_env = summarize_env_by_dominant_class(df_nan, frequency=frequency, dominant_frac_threshold=dominant_frac_threshold)
    
    # Define Wind direction bins into 8 and 16 compass sectors
    direction_labels_8=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    direction_labels_16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',]
    
    df_env = get_wind_sectors(df_env, direction_labels=direction_labels_8, )
    df_env = get_wind_sectors(df_env, direction_labels=direction_labels_16, col_out_name='compass')
    
    # Define a lookup table
    data_max_map = {
        (None, None): (2500, 6.5),
        (None, '1min'): (500, 5),
        (None, '10min'): (150, 6.5),
        (None, '30min'): (70, 10),
        (0.2, None): (2500, 6.5),
        (0.2, '1min'): (700, 6.5),
        (0.2, '10min'): (150, 6.5),
        (0.2, '30min'): (70, 10),
        (0.5, None): (2500, 6.5),
        (0.5, '1min'): (700, 6.5),
        (0.5, '10min'): (150, 6.5),
        (0.5, '30min'): (70, 14),
        (0.6, None): (2500, 6.5),
        (0.6, '1min'): (500, 7),
        (0.6, '10min'): (150, 6.5),
        (0.6, '30min'): (70, 14),
        (0.7, None): (2500, 6.5),
        (0.7, '1min'): (600, 8),
        (0.7, '10min'): (100, 14),
        (0.7, '30min'): (70, 14)
    }
    # Look up the value
    data_max = data_max_map.get((dominant_frac_threshold, frequency), None)
    # handle missing combinations
    if data_max is None:
        raise ValueError(f"No data_max defined for threshold={dominant_frac_threshold} and frequency='{frequency}'")

    
    # make figure directory to put in 
    try:
        new_dir = pathlib.Path(f"{fig_dir}/sf_class_frequ/{frequency}_{dominant_frac_threshold}/")
        new_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    
    
        # Normalized frequencies
    plt_snowflake_heatmap_by_sector(df_env, mode="percent", title=f"Relative Frequency of Snowflake Classes by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", cmap_base=cm.managua)
    figname = f"{new_dir}/relative_sf_class_frequency_by_wind_sector.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)

    

    # Absolute counts
    plt_snowflake_heatmap_by_sector(df_env, mode="absolute", title=f"Absolute Count of Snowflake Classes by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", data_max=data_max[0], cmap_base=cm.managua)
    figname = f"{new_dir}/absolute_count_sf_class_frequency_by_wind_sector.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)


    # Graupel to Aggregate ratio
    plt_snowflake_heatmap_by_sector(df_env, mode="ratio", class_ratio=("graupel", "aggregate"), title = f"Graupel-to-Aggregate Ratio by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", figsize=(10, 2), data_max=data_max[1],cmap_base=cm.managua)
    figname = f"{new_dir}/graupel_to_aggregate_ratio_by_wind_sector.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)


    #Group counts Compute Snowflake Frequencies per ( Wind Sector)
    counts = df_env.groupby(['wind_direction_sector', 'snowflake_class_name']).size().reset_index(name='count')
    # Pivot to a contingency table: rows=(temp, wind), columns=snowflake_class
    pivot_table = counts.pivot_table(index=['wind_direction_sector'], columns='snowflake_class_name', values='count', fill_value=0)


    plt_dominant_snowflake_class_by_sector(pivot_table, class_colors=custom_colors, title=f"Dominant Snowflake Class by Wind Sector ({frequency} - ≥{dominant_frac_threshold})")
    figname = f"{new_dir}/bar_dominant_sf_class_by_wind_sector.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)

    plt_dominant_snowflake_class_heatmap(pivot_table, class_colors=custom_colors, title=f"Dominant Snowflake Class by Wind Sector (Heatmap) ({frequency} - ≥{dominant_frac_threshold})")
    figname = f"{new_dir}/heatmap_dominant_sf_class_by_wind_sector.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)

    # plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_8, class_color=custom_colors, title=f"Snowflake Types by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", mode="per_direction")
    
    plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_16, class_color=custom_colors, title=f"Snowflake Types by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", mode="per_direction")
    
    # plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_8, class_color=custom_colors, title=f"Snowflake Types by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", mode="count")
    
    plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_16, class_color=custom_colors, title=f"Snowflake Types by Wind Sector ({frequency} - ≥{dominant_frac_threshold})", mode="count")
    figname = f"{new_dir}/windrose_sf_class_by_wind_sector_count.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)

    # plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_8, class_color=custom_colors, title=f"Snowflake Types by Wind sector - Normalized ({frequency} - ≥{dominant_frac_threshold})", mode="total")

    plt_stacked_snowflake_windrose(df_env, direction_labels=direction_labels_16, class_color=custom_colors, title=f"Snowflake Types by Wind sector - Normalized ({frequency} - ≥{dominant_frac_threshold})", mode="total")
    figname = f"{new_dir}/windrose_sf_class_by_wind_sector_normalized_total.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)
    
    plt_timeseries_temp_snowflake_class(df_env, class_color=custom_colors, title=f"Snowflake Type Temperature ({frequency} - ≥{dominant_frac_threshold})")
    figname= f"{new_dir}/timeseries_temp_snowflake_class.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)
    
    
    plt_timeseries_wind_snowflake_class(df_env, class_color=custom_colors, title=f"Snowflake Type Wind Direction ({frequency} - ≥{dominant_frac_threshold})")
    figname= f"{new_dir}/timeseries_wd_snowflake_class.png"
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True,dpi=300)
    
    return df_env