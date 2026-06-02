
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




site = "ANX"
year = "2023"
version = IMProToo.__version__
base_height = np.arange(100, 3200, 100)  # Predefined MRR profile


filename_template = "Z_S_2Tm_WD_WS_class_precip_event_dt_offset_{site}_{year}.nc"


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
    


# Fill values to minuetly data 
def align_variable(source, target):
    if source is None:
        return target.copy(deep=True)
    return fct.arrays_same_time_res(source, target)


# load only what I need for ONE day
def load_day_slice(file_list, var, day_start, day_end):
    for f in file_list:
        ds = xr.open_dataset(f, engine="h5netcdf", cache=False)
        if var in ds:
            try:
                out = ds[var].sel(time=slice(day_start, day_end))
                if out.time.size > 0:
                    out.load()
                    return out

            finally:
                ds.close()
    return out


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
        del ds_mrr, Sret, S_ensemble #, dsx
        gc.collect()





# Accumulate incrementally
def extract_event_tables(dsx, a, b):
    df_sum = dsx[['precipitation', 'Matrosov_aggregate_400m', 'Bracci_aggregate_400m', 'Nakai_aggregate_400m', 'Nakai_graupel_400m', 'Pu_graupel_400m', 'Schoger_400m', f'S_{a}_{b}_400m',]].to_dataframe().sort_values("time")



    df_mean = dsx[['air_temperature', 'wind_direction', 'wind_speed', 'W_400m']].to_dataframe().sort_values("time")


    return df_sum, df_mean


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


# Time offset
def apply_offset(df_corr, df_sum, df_mean, a, b):
    df_off = fct.apply_timeoffset_to_df(df=df_corr, df_sum_offset=df_sum, df_mean_offset=df_mean, a=a, b=b)
    df_off.index.name = "time"
    return df_off


# Event processing
def run_event_processing(df_sum, df_mean, sf_class, a,b):
    df_corr, psd_corr = fct.process_precip_segments(df_sum, df_mean, sf_class, mode="event", a=a, b=b)

    df_corr.index.name = "time"
    return df_corr, psd_corr


def write_final(ds, out_path):
    encoding = {v: {"zlib": True, "complevel":4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding, engine="h5netcdf")
    ds.close()


def run_nearest_crystal_variant(df_sum, df_mean, sf_class):
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

    
    df_corr, psd_corr = run_event_processing(t_df_sum, t_df_mean, sf_assigned, a, b)

    t_df_sum_off = t_df_sum[['Matrosov_aggregate_400m', 'Bracci_aggregate_400m', 'Nakai_aggregate_400m', 'Nakai_graupel_400m', 'Pu_graupel_400m', 'Schoger_400m', f'S_{a}_{b}_400m']]
    t_df_mean_off = t_df_mean[['W_400m']]
    
    df_corr_offset = apply_offset(df_corr, t_df_sum_off, t_df_mean_off, a, b)

    return df_corr_offset.to_xarray()



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


a_int = int(80)
b_val = round(1.0,2)
a= a_int
b=b_val





def process_ensemble_member(a, b, INPUT_DATA_DIR, OUTPUT_DATA_DIR, site, year, version):
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
    df_corr, psd_corr = run_event_processing(df_sum, df_mean, sf_class, a, b)
    
    



    # ------------------------------------------------------------------
    # 5. Apply time offset (fall speed)
    # ------------------------------------------------------------------
    # print("🕒 Applying time offset")
    # assert df_corr.index.is_monotonic_increasing

    df_sum_offset = df_sum[['Matrosov_aggregate_400m', 'Bracci_aggregate_400m', 'Nakai_aggregate_400m', 'Nakai_graupel_400m', 'Pu_graupel_400m', 'Schoger_400m', f'S_{a}_{b}_400m']].copy()
    df_mean_offset = df_mean[['W_400m']].copy()
    
    df_corr_offset = apply_offset(df_corr, df_sum_offset, df_mean_offset, a, b)
    del df_corr, psd_corr
    gc.collect()
    
    # ------------------------------------------------------------------
    # 6. Merge & write FINAL product (event-based)
    # ------------------------------------------------------------------

    # print(df_corr_offset.columns[df_corr_offset.columns.duplicated()])

    ds_final = df_corr_offset.to_xarray()

    del df_corr_offset
    gc.collect()


    out_file = os.path.join(nc_dir, f"Z_S_2Tm_WD_WS_class_precip_event_dt_offset_{site}_{year}.nc")
    if len(ds_final.time) > 0:
        write_final(ds_final, out_file)
        print(f"Written {out_file}")


    # # ------------------------------------------------------------------
    # # 7. Nearest-crystal variant (optional but faithful)
    # # ------------------------------------------------------------------
    # ds_final_nearest = run_nearest_crystal_variant(df_sum, df_mean, sf_class)
    # out_file = os.path.join(nc_dir, f"Z_S_2Tm_WD_WS_nearest_class_precip_event_dt_offset_{site}_{year}.nc")
    # if len(ds_final_nearest.time) > 0:
    #     write_final(ds_final, out_file)
    #     print(f"Written {out_file}")

    # # ------------------------------------------------------------------
    # 10. Cleanup (IMPORTANT)
    # ------------------------------------------------------------------
    del df_sum, df_mean, df_sum_offset, df_mean_offset, sf_class, ds_final, #ds_final_nearest
    gc.collect()


    


if __name__ == "__main__":
    # a_range = np.concatenate([np.arange(1,10,1), np.arange(10,330,10)])
    a_range = np.arange(300,330,10)
    b_range = np.arange(5, 6.1, 0.1)#np.arange(0.8,6.1,0.1)


    for a in a_range:
        for b in b_range:
            start_time = time.perf_counter()
            a = int(a)
            b = round(b, 2)
            
            out_name = f'{OUTPUT_DATA_DIR}/ANX_obs_EM_{a}_{b}/Z_S_2Tm_WD_WS_class_threshold_precip_event_dt_offset_{site}_{year}.nc'
            if out_name not in glob(out_name):
                print(f"\n🔹 Running ensemble member a={a}, b={b}")
                process_ensemble_member(a, b, INPUT_DATA_DIR, OUTPUT_DATA_DIR, site, year, version)
                end_time = time.perf_counter()

                elapsed_time = round((end_time - start_time), 2)
                print(f"Execution time (a={a}, b={b}): {elapsed_time} s")

















        # # cleanup_intermediate_files(a, b)














