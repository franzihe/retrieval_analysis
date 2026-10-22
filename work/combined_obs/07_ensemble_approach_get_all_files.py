# %%
import os
import pathlib
import sys
import socket
import numpy as np
import IMProToo
import xarray as xr
from glob import glob
import gc
import shutil
import psutil
import argparse


hostname = socket.gethostname()
print(hostname)


# abs_path = str(pathlib.Path(hostname).parent.absolute())
abs_path = str(pathlib.Path(hostname).parent.absolute())

WORKDIR = abs_path[:- (len(abs_path.split('/')[-2] + abs_path.split('/')[-1])+1)]
DATA_DIR = "/mn/vann/franzihe/"


INPUT_DATA_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'output')
UTILS_DIR = os.path.join(WORKDIR, 'utils')

sys.path.append(UTILS_DIR)
from imports import pd, time
import functions as fct



# %%
site = "ANX"
year = "2023"
version = IMProToo.__version__
base_height = np.arange(100, 3200, 100)  # Predefined MRR profile

# %%


# %%
def process_single_file_mrr(fname, a, b, site, version, out_dir):
    date_tag = pathlib.Path(fname).stem[-8:]
    out_name = os.path.join(out_dir, f"mrr_improtoo_{version}_{site}_{date_tag}.nc")

    if os.path.exists(out_name):
        return out_name

    # Open dataset safely with dask chunks
    # with xr.open_dataset(fname, chunks={"time": 100}) as ds:
    with xr.open_dataset(fname, engine="h5netcdf") as ds:
        ds = ds.load()
    
    try:
        # Optional: drop variables not needed
        drop_vars = ['Matrosov_aggregate', 'Bracci_aggregate', 'Nakai_aggregate',
                     'Nakai_graupel', 'Pu_graupel']
        ds = ds.drop_vars([v for v in drop_vars if v in ds])

        # Noise-correct Ze
        ds['Ze'] = fct.noise_corrected_ze(ds['Ze'])

        # Compute Z-S
        df_snowfall_rate = fct.ze_to_snowfall_rate(ds['Ze'], a, b)
        ds = xr.merge([ds, df_snowfall_rate.to_xarray()])

        # Assign attributes for new variables
        for name in ['Matrosov_aggregate', 'Bracci_aggregate', 'Nakai_aggregate',
                     'Nakai_graupel', 'Pu_graupel', 'Schoger', f'S_{a}_{b}']:
            if name in ds:
                ds[name].attrs.update(units='mm/h')

        # Save dataset safely
        ds.to_netcdf(out_name, engine="h5netcdf")
    finally:
        ds.close()
        del ds, df_snowfall_rate
        gc.collect()
    return out_name

# %%
def list_input_files(INPUT_DATA_DIR, site, year, a, b):
    # Collect file paths

    mrr    = sorted(glob(os.path.join(f"{INPUT_DATA_DIR}/MRR_ANX/maahn/mrr_improtoo*_{site}_{year}*.nc")))
    met = {
        "temp": sorted(glob(f"{INPUT_DATA_DIR}/hatpro/hatpro_new_20240716/level1/{year}/air_temperature_{year}*.nc", recursive=True)),
        "wind": sorted(glob(f"{INPUT_DATA_DIR}/Met-No_obs/SN87110/ANX_wind_10min_*{year}.nc")),
        "prec": sorted(glob(f"{INPUT_DATA_DIR}/geonor/precip_{year}*.nc")),
    }
    sf_class = sorted(glob(os.path.join(f"{INPUT_DATA_DIR}/MASC_DB/Norway-{year}_MASCDB_aggregate_graupel_pristine.nc")))

    return mrr, met, sf_class


# %%
def map_mrr_files_by_date(mrr_files):
    """
    Map YYYYMMDD → MRR file path
    """
    out = {}
    for f in mrr_files:
        stem = pathlib.Path(f).stem
        date = stem[-8:]
        out[date] = f
    return out

# %%
# create the empty daily dataset
def make_empty_day_dataset(time, range_vals):
    empty1D = np.empty(time.shape,)
    empty1D[:] = np.nan
    empty2D = np.empty((time.shape[0], range_vals.shape[0]))
    empty2D[:,:] = np.nan


    return xr.Dataset(coords=dict(time=time, range=range_vals),
                      data_vars=dict(
                          air_temperature=(["time"], empty1D),
                          wind_direction =(["time"], empty1D),
                          wind_speed=(["time"], empty1D),
                          precipitation=(["time"], empty1D),
                          Ze=(["time","range"], empty2D),
                          Ze_linear=(["time","range"], empty2D),
                          W=(["time","range"], empty2D),
                          spectralWidth=(["time","range"], empty2D),
                          Matrosov_aggregate=(["time","range"], empty2D),
                          Bracci_aggregate=(["time","range"], empty2D),
                          Nakai_aggregate=(["time","range"], empty2D),
                          Nakai_graupel=(["time","range"], empty2D),
                          Pu_graupel=(["time","range"], empty2D),
                          Schoger=(["time","range"], empty2D),
                      ))
    

# %%
# Fill values to minuetly data 
def align_variable(source, target):
    if source is None:
        return target.copy(deep=True)
    return fct.arrays_same_time_res(source, target)

# %%
# load only what I need for ONE day
def load_day_slice(file_list, var, day_start, day_end):
    for f in file_list:
        with xr.open_dataset(
            f, engine="h5netcdf", cache=False
        ) as ds:
            if var not in ds:
                continue

            out = ds[var].sel(time=slice(day_start, day_end))

            if out.time.size > 0:
                return out.load()

    return None

# %%
# per day processing
def process_one_day(date, mrr_files, met_files,  a, b,):#year, base_height, out_dir, site, ):
    # Define start and end time of individual day
    day_start = pd.Timestamp(f"{date} 00:00:00")
    day_end = day_start + pd.Timedelta(hours=23, minutes=59)
    time = pd.date_range(start=day_start, end=day_end, freq='1min')

    # --- load MRR for day
    ds_mrr = xr.open_dataset(mrr_files[date], engine="h5netcdf", cache=False)[["Ze","W","spectralWidth", "Matrosov_aggregate","Bracci_aggregate","Nakai_aggregate","Nakai_graupel","Pu_graupel","Schoger"]]
    ds_mrr = ds_mrr.sel(time=slice(day_start, day_end))
    ds_mrr.load()

    try: 
        dsx = make_empty_day_dataset(time, ds_mrr.range.values)
        
        # --- met vars ---
        dsx["air_temperature"] = align_variable(load_day_slice(met_files["temp"], "air_temperature", day_start, day_end), dsx["air_temperature"])

        dsx["wind_direction"] = align_variable(load_day_slice(met_files["wind"], "wind_direction_filled", day_start, day_end), dsx["wind_direction"])

        dsx["wind_speed"] = align_variable(load_day_slice(met_files["wind"], "wind_speed_filled", day_start, day_end), dsx["wind_speed"])

        dsx["precipitation"] = align_variable(
            load_day_slice(met_files["prec"], "precipitation", day_start, day_end),
            dsx["precipitation"])
        
        # --- radar ---
        Ze = ds_mrr["Ze"].where(ds_mrr["Ze"] >= -20.)
        dsx["Ze"] = align_variable(Ze, dsx["Ze"])

        dsx["Ze_linear"] = 10**(dsx["Ze"]/10)

        W = (ds_mrr["W"].where(ds_mrr["Ze"] >= -20.))
        dsx["W"] = align_variable(W, dsx["W"])

        sWidth = ds_mrr["spectralWidth"].where(ds_mrr["Ze"] >= -20.)
        dsx["spectralWidth"] = align_variable(sWidth, dsx["spectralWidth"])

        

        for name in ['Matrosov_aggregate', 'Bracci_aggregate', 'Nakai_aggregate', 'Nakai_graupel', 'Pu_graupel', 'Schoger']:
            Sret = ds_mrr[f"{name}"].where(ds_mrr["Ze"] >= -20.)
            dsx[f"{name}"] = align_variable(Sret, dsx[f"{name}"])

        # --- calculate ensemble member 
        S_ensemble = (dsx["Ze_linear"]/a) **(1/b)
        S_ensemble = (S_ensemble/60.)
        dsx[f"S_{a}_{b}"] = S_ensemble.rename(f"S_{a}_{b}")
            

        # --- altitude extraction ---
        # rbin_map = {"400": 3, "800": 7, "1000": 9}
        # for alt, rbin in rbin_map.items():
        #     for name in var_names:
        #         dsx[f"{name}_{alt}m"] = dsx[name].isel(range=rbin)
        alt = 400
        rbin = 3
        for name in [f"Ze", "W", 'Matrosov_aggregate', 'Bracci_aggregate', 'Nakai_aggregate','Nakai_graupel', 'Pu_graupel', 'Schoger', f'S_{a}_{b}']:
                dsx[f"{name}_{alt}m"] = dsx[f"{name}"].isel(range=rbin).drop_vars("range")

        
        return dsx
    finally:
        ds_mrr.close()
        # dsx.close()
        # del ds_mrr, Sret, S_ensemble #, dsx
        gc.collect()

# %%


# %%
# Accumulate incrementally
def extract_event_tables(dsx, a, b):
    df_sum = dsx[['precipitation', 'Matrosov_aggregate_400m', 'Bracci_aggregate_400m', 'Nakai_aggregate_400m', 'Nakai_graupel_400m', 'Pu_graupel_400m', 'Schoger_400m', f'S_{a}_{b}_400m',]].to_dataframe().sort_values("time")



    df_mean = dsx[['air_temperature', 'wind_direction', 'wind_speed', 'W_400m']].to_dataframe().sort_values("time")


    return df_sum, df_mean

# %%
def build_season_tables(mrr_map, met_files, a, b):
    dfs_sum = []
    dfs_mean = []
    dsx_all = []
    for date in mrr_map:
        dsx = process_one_day(date, mrr_map, met_files, a, b)
        dsx_all.append(dsx)

        df_sum, df_mean = extract_event_tables(dsx, a, b)
        dfs_sum.append(df_sum)
        dfs_mean.append(df_mean)

        dsx.close()

    return(
        xr.concat(dsx_all, dim="time").sortby("time"),
        pd.concat(dfs_sum).sort_index(),
        pd.concat(dfs_mean).sort_index()
        )

# %%
def calculate_class_fallspeeds(df_windows, df_mean, duration, class_column="snowflake_class_name"):
    """
    Calculate class-specific mean MRR fall speeds using all valid
    original 1-minute observations within selected fixed windows.
    """

    # Select windows passing Geonor and classification criteria
    selected = df_windows.loc[
        df_windows["valid_precip"]
        & df_windows[class_column].notna()
    ].copy()

    # Assign original MRR observations to window ends
    mrr = df_mean[["W_400m"]].copy()
    mrr = mrr.loc[
        (mrr["W_400m"] >= -7)
        & (mrr["W_400m"] <= 7)].copy()

    mrr["window_end"] = (
        mrr.index.floor(duration)
        + pd.Timedelta(duration)
    )

    # Match MRR observations to selected windows
    mrr = mrr.join(
        selected[[class_column]],
        on="window_end",
        how="inner"
    ).dropna(subset=["W_400m"]).copy()

    # Calculate class-specific statistics
    stats = (
        mrr.groupby(class_column)["W_400m"]
        .agg(["mean", "std", "count"])
    )

    stats["n_windows"] = (
        mrr.groupby(class_column)["window_end"]
        .nunique()
    )

    return stats, mrr

# %%
# Time offset
def apply_offset(df_corr, df_sum, df_mean, a, b, mode="fixed", duration="4min"):
    df_corr = df_corr.copy()
    class_column = f"snowflake_class_name"

    if mode == "fixed":
        # calculate constant fall speed from MRR for each class
        stats_fixed, _ = calculate_class_fallspeeds(df_corr, df_mean, duration, class_column=class_column)
        # assign constant fall speed to each classified window
        class_fallspeeds = stats_fixed['mean']
        # df_corr['W_class_mean'] = (df_corr[class_column].map(class_fallspeeds).where(df_corr['W_400m'].notna()))
        df_corr["W_class_mean"] = (
            df_corr[class_column].map(class_fallspeeds)
            )   # calculate one mean velocity for each snowflake class and apply that velocity to all windows belonging to that class. means that windows without an unshifted MRR velocity can still receive a time offset based on their MASC snowflake classification.

    # now apply time offset    
    df_off = fct.apply_timeoffset_to_df(df=df_corr, df_sum_offset=df_sum, df_mean_offset=df_mean, a=a, b=b, mode=mode)
    df_off.index.name = "time"

    # Select valid observations after time correction
    valid = (
        (df_off["rate_mm_h"] >= 0.1)    # Valid Geonor precipitation
        & df_off[class_column].notna()  # MASC classification at the selected threshold
        # & df_off["W_400m"].notna()      # MRR velocity available in the shifted interval
    )
    if mode == "fixed":
        valid &= df_off["W_class_mean"].gt(0)  # Valid positive class-mean velocity for calculating the fall time
    elif mode == "event":
        valid &= df_off['fallspeed'].gt(0)

    df_off = df_off.loc[valid].copy()
    return df_off

# %%
def run_event_processing(
    df_sum, df_mean, sf_class, a, b,
    mode="fixed", duration="4min", threshold=60
):
    df_corr, psd_corr = fct.process_precip_segments(
        df_sum,
        df_mean,
        sf_class,
        mode=mode,
        duration=duration,
        a=a,
        b=b,
        threshold=threshold
    )

    df_corr.index.name = "time"
    return df_corr, psd_corr

# %%
def write_final(ds, out_path):
    encoding = {v: {"zlib": True, "complevel":4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding, engine="h5netcdf")
    ds.close()

# %%
def run_nearest_crystal_variant(df_sum, df_mean, sf_class, mode="event", duration="1H", threshold=60):
    t_df_sum = df_sum.copy()
    t_df_mean = df_mean.copy()
    t_sf_class = sf_class.copy()

    # Step 1 - round crystal timestamps to the nearest minute
    t_sf_class["minute"] = t_sf_class.index.floor("min")
    sf_groups = t_sf_class.groupby("minute")
    available_minutes = pd.DatetimeIndex(list(sf_groups.groups.keys()))
    # Step 2 - for each Z-S minute, find crystals
    window = pd.Timedelta("15min")
    assigned_rows =[]

    for t in t_df_sum.index:
        if t in sf_groups.groups:
            rows = sf_groups.get_group(t)
            rows_copy = rows.copy()
            rows_copy["original_time"] = rows_copy.index
            rows_copy["assigned_time"] = t
            assigned_rows.append(rows_copy)
        else:
            diffs = np.abs(available_minutes -t)
            inside_window = diffs <= window
            if inside_window.any():
                # pick nearest minute that has cyrstals
                nearest_minute = available_minutes[inside_window][diffs[inside_window].argmin()]
                rows = sf_groups.get_group(nearest_minute)
                rows_copy = rows.copy()
                rows_copy["original_time"] = rows_copy.index
                rows_copy["assigned_time"] = t
                assigned_rows.append(rows_copy)
    # Step 3 - combine everything
    sf_assigned = pd.concat(assigned_rows, ignore_index=False)
    sf_assigned = sf_assigned.set_index("assigned_time").sort_index()

    
    df_corr, psd_corr = run_event_processing(t_df_sum, t_df_mean, sf_assigned, a, b, mode=mode, duration=duration, threshold=threshold)

    t_df_sum_off = t_df_sum[['Matrosov_aggregate_400m', 'Bracci_aggregate_400m', 'Nakai_aggregate_400m', 'Nakai_graupel_400m', 'Pu_graupel_400m', 'Schoger_400m', f'S_{a}_{b}_400m']]
    t_df_mean_off = t_df_mean[['W_400m']]
    
    df_corr_offset = apply_offset(df_corr, t_df_sum_off, t_df_mean_off, a, b, mode, duration)

    return df_corr_offset.to_xarray()


# %%
def cleanup_intermediate_files(a, b, base_dir):
    """
    Remove intermediate daily files for one ensemble member.

    This should ONLY be called after final products exist.
    """
    tmp_dir = (pathlib.Path(base_dir)/f"EM_{a}_{b}")

    if not tmp_dir.exists():
        print("No intermediate files to clean up.")
        return
    
    if tmp_dir.exists():
        print(f"🧹 Removing intermediate files in {tmp_dir}")
        shutil.rmtree(tmp_dir)

# %%
def final_output_path(nc_dir, site, year, mode, threshold, duration=None):
    label = f"{mode}_{duration}" if mode == "fixed" else "event"

    return os.path.join(
        nc_dir,
        f"{label}_{threshold}_dt_offset_{site}_{year}.nc"
    )

# %%
a_int = int(80)
b_val = round(1.0,2)
a= a_int
b=b_val

# %%


# %%
def process_ensemble_member(a, b, INPUT_DATA_DIR, OUTPUT_DATA_DIR, site, year, version, out_file, mode="fixed", duration="4min", threshold=60):
    print("\n" + "="*60)
    print(f"🚀 START ensemble member a={a}, b={b}")
    print(f"Python PID: {os.getpid()}")
    print(f"🧠 RAM usage: {psutil.virtual_memory().percent:.1f}%")


    # -------------------------
    # Clean memory
    # -------------------------
    gc.collect()
    # print(f"🧠 RAM usage: {psutil.virtual_memory().percent:.1f}%")

    # -------------------------
    # Create directories
    # -------------------------

    nc_dir = pathlib.Path(os.path.join(OUTPUT_DATA_DIR, f'ANX_obs_EM_{a}_{b}'))
    nc_dir.mkdir(parents=True, exist_ok=True)



    # -------------------------
    # 1. Locate files
    # -------------------------
    file_pattern = f"{INPUT_DATA_DIR}/MRR_ANX/maahn/mrr_improtoo_{version}_{site}_*{year}*.nc"
    mrr_files = sorted(glob(file_pattern))

    # print(f"📁 Found {len(mrr_files)} raw MRR files")
    # assert len(mrr_files) > 0, "No input MRR files found"

    mrr_files, met_files, sf_files = list_input_files(INPUT_DATA_DIR, site, year, a, b)

    mrr_map = map_mrr_files_by_date(mrr_files)
    # print(f"📅 MRR days available: {len(mrr_map)}")
    # assert len(mrr_map) > 0, "No processed MRR files found"
    # example_date = next(iter(mrr_map))
    # print("Example MRR file:", example_date, mrr_map[example_date])



    # ------------------------------------------------------------------
    # 2. Build season-wide event tables (STREAMING)
    # ------------------------------------------------------------------
    dsx, df_sum, df_mean = build_season_tables(mrr_map, met_files,  a, b)
    # out_file = os.path.join(nc_dir, f"10min_filled_obs_hatpro_airport_WD_WS_T_{site}_{year}.nc")
    # if len(dsx.time) > 0:
    #     write_final(dsx, out_file)
    #     print(f"Written {out_file}")
    gc.collect()
    



    # ------------------------------------------------------------------
    # 3. Load snowflake class database (ONCE)
    # ------------------------------------------------------------------
    with xr.open_dataset(sf_files[0], engine="h5netcdf", cache=False, decode_timedelta=True) as sf_class:
        sf_class = sf_class[["snowflake_class_name", "fallspeed", "flake_Dmax", "gan3d_mass"]].to_dataframe().sort_index()
    # sf_class = xr.open_dataset(sf_files[0], engine="h5netcdf", cache=False, decode_timedelta=True)[["snowflake_class_name", "fallspeed", "flake_Dmax", "gan3d_mass"]].to_dataframe().sort_index()

    # print("❄️ snowflake class size:", sf_class.shape)
    # assert not sf_class.index.duplicated().any()


    # Limit fallspeed
    df_mean.loc[df_mean['W_400m'] > 7.0, 'W_400m'] = np.nan
    df_mean.loc[df_mean['W_400m'] <= -7.0, 'W_400m'] = np.nan

    # print("🌬 W_400m valid fraction:",
    #   df_mean['W_400m'].notna().mean())


    sf_class.loc[sf_class['fallspeed'] > 7., 'fallspeed'] = np.nan
    sf_class.loc[sf_class['fallspeed'] <= 0.2, 'fallspeed'] = np.nan
    sf_class = sf_class[sf_class['fallspeed'].notna()]

    # ------------------------------------------------------------------
    # 4. Process precipitation events
    # ------------------------------------------------------------------
    # print("⏱ Running event processing")
    # print("df_sum shape:", df_sum.shape)
    # print("df_mean shape:", df_mean.shape)
    df_corr, psd_corr = run_event_processing(df_sum, df_mean, sf_class, a, b, mode=mode, duration=duration, threshold=threshold)
    
    



    # ------------------------------------------------------------------
    # 5. Apply time offset (fall speed)
    # ------------------------------------------------------------------
    # print("🕒 Applying time offset")
    # assert df_corr.index.is_monotonic_increasing

    df_sum_offset = df_sum[
        [
            'Matrosov_aggregate_400m', 
            'Bracci_aggregate_400m', 
            'Nakai_aggregate_400m', 
            'Nakai_graupel_400m', 
            'Pu_graupel_400m', 
            'Schoger_400m', 
            f'S_{a}_{b}_400m'
            ]
            ].copy()
    df_mean_offset = df_mean[['W_400m']].copy()

    df_corr_offset = apply_offset(
            df_corr, 
            df_sum_offset, 
            df_mean_offset, 
            a, 
            b, 
            mode=mode, 
            duration=duration, )
    print(
        f"{threshold}%: "
        f"{len(df_corr_offset)} observations"
    )
    # coverage = df_corr_offset["radar_coverage"]

    # print(f"\nRadar coverage: {mode}, {duration}")
    # print(f"Total retained windows: {len(df_corr_offset)}")
    # print(f"Complete windows (100%): {(coverage == 1).sum()}")
    # print(f"Complete windows (%): {(coverage == 1).mean() * 100:.1f}")
    # print(f"Windows >= 90% coverage: {(coverage >= 0.9).sum()}")
    # print(f"Windows >= 75% coverage: {(coverage >= 0.75).sum()}")
    # print(f"Mean radar coverage: {coverage.mean() * 100:.1f}%")
    # print(f"Minimum radar coverage: {coverage.min() * 100:.1f}%")
    # print(df_corr_offset["radar_coverage"]
    #       .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    #       )
    # print("Coverage > 100%:", (df_corr_offset["radar_coverage"] > 1.001).sum())

    # print("No radar data:", (coverage == 0).sum())
    # print("Partial coverage:", ((coverage > 0) & (coverage < 1)).sum())
    # print("Complete coverage:", (coverage == 1).sum())

    del df_corr, psd_corr
    gc.collect()
    
    # ------------------------------------------------------------------
    # 6. Merge & write FINAL product (event-based)
    # ------------------------------------------------------------------

    # print(df_corr_offset.columns[df_corr_offset.columns.duplicated()])
    df_final = df_corr_offset.copy()

    # # Keep only the classification column for this threshold
    # other_thresholds = [t for t in [60, 70, 80] if t != threshold]

    # columns_to_drop = [
    #         f"snowflake_class_name_{t}"
    #         for t in other_thresholds
    #         if f"snowflake_class_name_{t}" in df_final.columns
    #     ]

        # df_final = df_final.drop(columns=columns_to_drop)

    ds_final = df_final.to_xarray()

    

    if ds_final.sizes.get("time", 0) > 0:
        write_final(ds_final, out_file)
        print(f"Written {out_file}")
    else:
        print(f"No observations for threshold {threshold}%")
        ds_final.close()

    del df_final, ds_final

    # # ------------------------------------------------------------------
    # # 7. Nearest-crystal variant (optional but faithful)
    # # ------------------------------------------------------------------
    # ds_final_nearest = run_nearest_crystal_variant(df_sum, df_mean, sf_class)
    # out_file = os.path.join(nc_dir, f"Z_S_2Tm_WD_WS_nearest_class_precip_event_dt_offset_{site}_{year}.nc")
    # if len(ds_final_nearest.time) > 0:
    #     write_final(ds_final, out_file)
    #     print(f"Written {out_file}")

    # ------------------------------------------------------------------
    # 10. Cleanup (IMPORTANT)
    # ------------------------------------------------------------------
    del df_corr_offset
    gc.collect()


    


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=int, required=True, choices=range(6))
args = parser.parse_args()

a_range = np.concatenate([
    np.arange(1,10,1), 
    np.arange(10,330,10)
    ])
b_range = np.arange(0.8,6.1,0.1)

# Split a_range into 6 non-overlapping groups
a_groups = np.array_split(a_range, 6)
a_range = a_groups[args.worker]

print(f"Worker {args.worker}: a values = {a_range}", flush=True)

duration = "10min"
# duration = None
mode = "fixed"
threshold = 60
for a in a_range:
    for b in b_range:
        a = int(a)
        b = round(b, 2)

        nc_dir = os.path.join(OUTPUT_DATA_DIR, f"ANX_obs_EM_{a}_{b}")

        # for threshold in [60, ]:#70, 80]:
        if duration == None:
                out_file = final_output_path(nc_dir, site, year, mode=mode, threshold=threshold) 
        else:
                out_file = final_output_path(nc_dir, site, year, mode=mode, threshold=threshold, duration=duration)
        if os.path.exists(out_file):
                print(f"Skipping a={a}, b={b}, threshold={threshold}%")
                continue
        start_time = time.perf_counter()

        print(
            f"\nWorker {args.worker}: running a={a}, b={b}",
            flush=True
        )
        if duration == None:
                process_ensemble_member(a, b, INPUT_DATA_DIR, OUTPUT_DATA_DIR, site, year, version, mode=mode, threshold=threshold, out_file=out_file) 
        else:
                process_ensemble_member(a, b, INPUT_DATA_DIR, OUTPUT_DATA_DIR, site, year, version, mode=mode, threshold=threshold, duration=duration, out_file=out_file)

        elapsed = time.perf_counter() - start_time
        print(
            f"Worker {args.worker}: a={a}, b={b} "
            f"finished in {elapsed:.1f} s",
            flush=True
        )


# %%



