# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cdsapi

import numpy as np
import xarray as xr

import dask
from dask.diagnostics import ProgressBar


dt_hours = 6  # in hours
min_pressure_level = 50
max_pressure_level = 1000
pressure_level_delta = 50
months = [f"{x+1:02d}" for x in range(0, 1)]

base_output_dir = os.path.join("latest_data")
if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)

year_dict = {
    "first": np.arange(1979, 1993),
    "second": np.arange(1993, 2006),
    "third": np.arange(2006, 2021),
    "all": np.arange(1979, 2021),
    "test": np.arange(2015, 2017),
}
years = year_dict["test"]
print("Years:", years)
print("Months:", months)

day_timings = [f"{x:02d}:00" for x in range(0, 24, dt_hours)]
print("Day timings:", day_timings)

pressure_levels = list(
    range(
        min_pressure_level,
        max_pressure_level + pressure_level_delta,
        pressure_level_delta,
    )
)
print("Pressure levels:", pressure_levels)

# Initialize the API
c = cdsapi.Client()

# Download the NetCDF files
file_list = []
for year in years:
    year_str = str(year)

    for month in months:
        month_str = month

        for pl in pressure_levels:
            file_str = os.path.join(
                base_output_dir, f"pl_{pl}_{year_str}-{month_str}.nc"
            )
            if not os.path.exists(file_str):
                print(f"Downloading {month_str}.{year_str} and pressure level {pl}")
                c.retrieve(
                    "reanalysis-era5-complete",
                    {
                        "class": "ea",
                        "expver": "1",
                        "levtype": "pl",
                        "stream": "oper",
                        "type": "an",
                        "grid": [0.25, 0.25],
                        "format": "netcdf",
                        "levelist": f"{pl}",
                        # u, v, w, z, t, q
                        "param": "131/132/135.128/129.128/130.128/133.128",
                        "date": f"{year_str}-{month_str}-01/to/{year_str}-{month_str}-31",
                        "time": day_timings,
                    },
                    file_str,
                )
            else:
                print(
                    f"Data for {month_str}.{year_str} and pressure level {pl} already exists!"
                )
            file_list.append(file_str)

        # we have only one pressure level here
        file_str = os.path.join(base_output_dir, f"sfc_{year_str}-{month_str}.nc")
        if not os.path.exists(file_str):
            c.retrieve(
                "reanalysis-era5-complete",
                {
                    "class": "ea",
                    "expver": "1",
                    "levtype": "sfc",
                    "stream": "oper",
                    "type": "an",
                    "grid": [0.25, 0.25],
                    "format": "netcdf",
                    "date": f"{year_str}-{month_str}-01/to/{year_str}-{month_str}-31",
                    "time": day_timings,
                    # 10u, 10v, 100u, 100v, 2t, sp, msl, tcvw
                    "param": "165.128/166.128/246.228/247.228/167.128/134.128/151.128/137.128",
                },
                file_str,
            )
        file_list.append(file_str)

# Convert the dataset into ZARR format
zarr_output_dir = os.path.join("latest_zarr")
if not os.path.isdir(zarr_output_dir):
    os.makedirs(zarr_output_dir)

zarr_output_files = []
for file in file_list:
    print("!! Loading file:", file)
    zarr_output_file = os.path.split(file.replace(".nc", ".zarr"))[1]
    zarr_output_file = os.path.join(zarr_output_dir, zarr_output_file)
    zarr_output_files.append(zarr_output_file)
    if os.path.exists(zarr_output_file):
        print("ZARR file already exists:", zarr_output_file)
        continue

    print("Creating ZARR output file from:", zarr_output_file)
    ds = xr.open_dataset(file)
    print(ds)
    ds.to_zarr(zarr_output_file, mode="w", consolidated=True, append_dim=None)
    del ds

# Concatenate all the files into a single dataset array
print("Concatenating ZARR output files...")
zarr_arrays = [xr.open_zarr(zarr_output_file) for zarr_output_file in zarr_output_files]
print(zarr_arrays)

era5_xarray = xr.concat(
    [xr.concat([z[x] for x in z.data_vars.keys()], dim="channel") for z in zarr_arrays],
    dim="channel",
)
era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
era5_xarray.name = "fields"
era5_xarray = era5_xarray.astype("float32")
np_arr = np.array(era5_xarray.values)
print("Dataset shape:", np_arr.shape)
assert not np.isnan(np_arr).any()

# Move the train and test
train_years = years[:-1]
test_years = years[-1:]
print(f"Train years: {train_years} / Test years: {test_years}")

hdf5_output_dir = os.path.join("latest_hdf5")
if not os.path.isdir(hdf5_output_dir):
    os.makedirs(hdf5_output_dir)

# Generate final HDF5 files
for year in years:
    # HDF5 filename
    split = (
        "train"
        if year in train_years
        else "test"
        if year in test_years
        else "out_of_sample"
    )
    hdf5_path = os.path.join(hdf5_output_dir, split)
    os.makedirs(hdf5_path, exist_ok=True)
    hdf5_path = os.path.join(hdf5_path, f"{year}.h5")

    # Save year using dask
    print(f"Saving {year} at {hdf5_path}")
    with dask.config.set(
        scheduler="threads",
        num_workers=8,
        threads_per_worker=2,
        **{"array.slicing.split_large_chunks": False},
    ):
        with ProgressBar():
            # Get data for the current year
            year_data = era5_xarray.sel(time=era5_xarray.time.dt.year == year)

            # Save data to a temporary local file
            year_data.to_netcdf(hdf5_path, engine="h5netcdf", compute=True)
    print(f"Finished Saving {year} at {hdf5_path}")
