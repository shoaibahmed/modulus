#!/bin/python


import os
import cdsapi

import numpy as np
import xarray as xr


output_file = "download.nc"
if not os.path.exists(output_file):
    print("Downloading ERA5 NetCDF file...")
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
                'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_combined_wind_waves_and_swell',
                'surface_pressure', 'total_precipitation',
            ],
            'year': '2016',
            'month': '01',
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'format': 'netcdf',
        },
        output_file
    )

zarr_output_file = "test.zarr"
if not os.path.exists(zarr_output_file):
    print("Creating ZARR output file from:", output_file)
    ds = xr.open_dataset(output_file)
    print(ds)
    ds.to_zarr(zarr_output_file, mode="w", consolidated=True, append_dim=None)
    del ds

print("Validating ZARR output file...")
ds = xr.open_zarr(zarr_output_file)
print(ds)
data_vars = list(ds.data_vars.keys())
print("Data var keys:", data_vars)
print("Null values in DS?", {var: bool(ds[var].isnull().any()) for var in data_vars})

# Each data var is a different variable -- concatenate them in the channel dim
era5_xarray = xr.concat(
    [ds[var] for var in data_vars], dim="channel"
)
era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
era5_xarray.name = "fields"
era5_xarray = era5_xarray.astype("float32")
np_arr = np.array(era5_xarray.values)
print("Dataset shape:", np_arr.shape)
assert not np.isnan(np_arr).any()
