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
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
pl_vars = ["z", "t", "u", "v", "r"]
non_pl_vars = ["10u", "10v", "100u", "100v", "2t", "sp", "msl", "tcvw"]
months = [f"{x+1:02d}" for x in range(0, 1)]

year_dict = {
    "first": np.arange(1979, 1993),
    "second": np.arange(1993, 2006),
    "third": np.arange(2006, 2021),
    "all": np.arange(1979, 2021),
    "test": np.arange(2015, 2017),
}

years = year_dict["test"]
day_timings = [f"{x:02d}:00" for x in range(0, 24, dt_hours)]

print("Years:", years)
print("Months:", months)
print("Day timings:", day_timings)
print("Pressure levels:", pressure_levels)

# u, v, w, z, t, q
# 131/132/135.128/129.128/130.128/133.128
# 10u, 10v, 100u, 100v, 2t, sp, msl, tcvw
# 165.128/166.128/246.228/247.228/167.128/134.128/151.128/137.128
# full list: https://github.com/coecms/era5/blob/master/era5/data/era5_vars.json
variable_to_code_dict = {
    "u": "131.128",
    "v": "132.128",
    "w": "135.128",
    "z": "129.128",
    "t": "130.128",
    "q": "133.128",
    "r": "157.128",
    "10u": "165.128",
    "10v": "166.128",
    "100u": "246.228",
    "100v": "247.228",
    "2t": "167.128",
    "sp": "134.128",
    "msl": "151.128",
    "tcvw": "137.128",
}

# Initialize the API
c = cdsapi.Client()

# Create the base directory
base_output_dir = os.path.join("73var_netcdf_data")
if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)

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
                        "param": "/".join([variable_to_code_dict[x] for x in pl_vars]),
                        "date": f"{year_str}-{month_str}-01/to/{year_str}-{month_str}-31",
                        "time": day_timings,
                    },
                    file_str,
                )
            else:
                print(
                    f"Data for {month_str}.{year_str} and pressure level {pl} already exists!"
                )
            file_list.append((file_str, year))

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
                    "param": "/".join([variable_to_code_dict[x] for x in non_pl_vars]),
                },
                file_str,
            )
        file_list.append((file_str, year))

# Create the ZARR output directory
zarr_output_dir = os.path.join("73var_zarr_data")
if not os.path.isdir(zarr_output_dir):
    os.makedirs(zarr_output_dir)

# Convert the dataset into ZARR format
zarr_output_files_dict = {}  # dictionary of years with all channel files in a year
for file, year in file_list:
    print("!! Loading file:", file)
    zarr_output_file = os.path.split(file.replace(".nc", ".zarr"))[1]
    zarr_output_file = os.path.join(zarr_output_dir, zarr_output_file)

    if year not in zarr_output_files_dict:
        zarr_output_files_dict[year] = []
    zarr_output_files_dict[year].append(zarr_output_file)

    if os.path.exists(zarr_output_file):
        print("ZARR file already exists:", zarr_output_file)
        continue

    print("Creating ZARR output file from:", zarr_output_file)
    ds = xr.open_dataset(file)
    print(ds)

    validate_nans = False
    if validate_nans:
        vars_with_nans = []
        for var_name, data_array in ds.data_vars.items():
            if data_array.isnull().any():
                vars_with_nans.append(var_name)
        if len(vars_with_nans) > 0:
            print(f'Variables containing NaN values: {vars_with_nans}')
            exit()

    # Specify the chunking options
    chunking = {"time": 1, "latitude": 721, "longitude": 1440}
    if "level" in ds.dims:
        chunking["level"] = 1

    # Re-chunk the dataset
    ds = ds.chunk(chunking)

    ds.to_zarr(zarr_output_file, mode="w", consolidated=True, append_dim=None)
    del ds

# Concatenate all the files into a single dataset array
print("Concatenating ZARR output files...")
era5_xarray_list = []
for key in zarr_output_files_dict:
    print(f"key: {key} / files: {len(zarr_output_files_dict[key])}")
    zarr_arrays = [xr.open_zarr(zarr_output_file) for zarr_output_file in zarr_output_files_dict[key]]

    # Concatenate all the variables in a file, as well as all the variables in a year
    # Expected size: (channel, time, lat, long)
    era5_xarray = xr.concat(
        [xr.concat([z[x] for x in z.data_vars.keys()], dim="channel") for z in zarr_arrays],
        dim="channel",
    )
    era5_xarray_list.append(era5_xarray)
    print(f"year: {key} / shape: {era5_xarray.shape}")

# Concatenate the variables for the different years in time dim
# Expected size: (channel, time * num_years, lat, long)
era5_xarray = xr.concat(
    era5_xarray_list,
    dim="time",
)
era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
era5_xarray.name = "fields"
era5_xarray = era5_xarray.astype("float32")
del era5_xarray_list
print("Full data shape:", era5_xarray.shape)

validate_nans = False
if validate_nans:
    print("Validating NaN error using conversion to numpy array!")
    np_arr = np.array(era5_xarray.values)
    print("Dataset shape:", np_arr.shape)
    assert not np.isnan(np_arr).any()

# Create the HDF5 output directory
hdf5_output_dir = os.path.join("73var_hdf5_data")
if not os.path.isdir(hdf5_output_dir):
    os.makedirs(hdf5_output_dir)

# Save mean and std
compute_mean_std = True
if compute_mean_std:
    stats_path = os.path.join(hdf5_output_dir, "stats")
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    means_output_file = os.path.join(stats_path, "global_means.npy")
    if not os.path.exists(means_output_file):
        print(f"Computing global mean...")
        era5_mean = np.array(
            era5_xarray.mean(dim=("time", "latitude", "longitude")).values
        )
        np.save(means_output_file, era5_mean.reshape(1, -1, 1, 1))
    else:
        print("Global mean file already exists!")

    std_output_file = os.path.join(stats_path, "global_stds.npy")
    if not os.path.exists(std_output_file):
        print(f"Computing global standard deviation...")
        era5_std = np.array(
            era5_xarray.std(dim=("time", "latitude", "longitude")).values
        )
        np.save(std_output_file, era5_std.reshape(1, -1, 1, 1))
    else:
        print("Global standard devaition file already exists!")

    print(f"Finished saving global mean and std at {stats_path}")

# Move the train and test
train_years = years[:-1]
test_years = years[-1:]
print(f"Train years: {train_years} / Test years: {test_years}")

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
    if os.path.exists(hdf5_path):
        print(f"{hdf5_path} already exists. Skipping file.")
        continue

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
            assert not year_data.isnull().any()

            # Save data to a temporary local file
            year_data.to_netcdf(hdf5_path, engine="h5netcdf", compute=True)
    print(f"Finished Saving {year} at {hdf5_path}")
