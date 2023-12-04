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


dt_hours = 6  # in hours
min_pressure_level = 50
max_pressure_level = 1000
pressure_level_delta = 50
months = [f"{x+1:02d}" for x in range(0, 12)]

base_path = os.path.join("data")
if not os.path.isdir(base_path):
    os.makedirs(base_path)
year_dict = {'first': np.arange(1979, 1993), 'second': np.arange(1993, 2006), 'third' : np.arange(2006, 2021),
             'all': np.arange(1979, 2021), 'test': np.arange(2015, 2016)}
years = year_dict['test']
print("Years:", years)
print("Months:", months)

day_timings = [f"{x:02d}:00" for x in range(0, 24, dt_hours)]
print("Day timings:", day_timings)

pressure_levels = list(range(min_pressure_level, max_pressure_level+pressure_level_delta, pressure_level_delta))
print("Pressure levels:", pressure_levels)

# Initialize the API
c = cdsapi.Client()

for year in years:
    year_str = str(year)

    for month in months:
        month_str = month

        for pl in pressure_levels:
            print(f"Downloading {month_str}.{year_str} and pressure level {pl}")

            file_str = os.path.join(base_path, f"pl_{pl}_{year_str}-{month_str}.nc")
            c.retrieve('reanalysis-era5-complete', 
                       {
                           'class': 'ea',
                           'expver': '1',
                           'levtype': 'pl',
                           'stream': 'oper',
                           'type': 'an',
                           'grid': [0.25, 0.25],
                           'format': 'netcdf',
                           'levelist': f'{pl}',
                           # u, v, w, z, t, q
                           "param": "131/132/135.128/129.128/130.128/133.128",
                           'date': f'{year_str}-{month_str}-01/to/{year_str}-{month_str}-31',
                           'time': day_timings,
                       },
                       file_str)

        # we have only one pressure level here
        file_str = os.path.join(base_path, f"sfc_{year_str}-{month_str}.nc")
        c.retrieve("reanalysis-era5-complete",
                   {
                       'class': 'ea',
                       'expver': '1',
                       'levtype': 'sfc',
                       'stream': 'oper',
                       'type': 'an',
                       'grid': [0.25, 0.25],
                       'format': 'netcdf',
                       'date': f'{year_str}-{month_str}-01/to/{year_str}-{month_str}-31',
                       'time': day_timings,     
                       # 10u, 10v, 100u, 100v, 2t, sp, msl, tcvw
                       "param": "165.128/166.128/246.228/247.228/167.128/134.128/151.128/137.128",
                   }, file_str)
