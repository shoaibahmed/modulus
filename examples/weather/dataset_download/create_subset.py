#!/bin/python

import os
import h5py


root_dir = "/lustre/fsw/sw_earth2_ml/test_datasets/73varQ/train/"
output_dir = "/lustre/fsw/nvresearch/ssiddiqui/era5_small/train/"

# Ensure the input / output directory exists
assert os.path.exists(root_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Output directory created:", output_dir)

copy_year = [2015, 2016]
for year in copy_year:
    target_file = os.path.join(root_dir, f"{year}.h5")
    assert os.path.exists(target_file)
    data_file = h5py.File(target_file, "r")
    print(f"year: {year} / data shape: {data_file['fields']}")

    # Take the data for the first month
    num_days = 31  # days in January
    dt = 6  # in hours
    num_vals_per_day = 24 // dt
    num_vals = num_days * num_vals_per_day
    print("Total vals:", num_vals)

    subset = data_file["fields"][:num_vals, :, :, :]
    print(f"year: {year} / subset size: {subset.shape}")

    # Open the new HDF5 file in write mode
    output_file_path = os.path.join(output_dir, f"{year}.h5")
    with h5py.File(output_file_path, "w") as output_file:  # Create a dataset in the new file and write the subset data
        output_file.create_dataset("fields", data=subset)
    print(f"Subset written to {output_file_path}")

    data_file.close()
