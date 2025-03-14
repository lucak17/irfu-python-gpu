#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from pyrfu import mms, pyrf
import timeit
import logging
import xarray as xr
import multiprocessing as mp


logging.disable(logging.CRITICAL)

from cupy.cuda import compiler


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__license__ = "MIT"

# Initialize MMS client
mms.db_init(default="local", local="./data/")

def split_dataset_by_time(ds, N):
    """
    Split an xarray Dataset along the 'time' dimension into N parts
    """
    total_times = ds.time.size
    # Compute indices that split the time dimension into N nearly equal parts.
    indices = np.array_split(np.arange(total_times), N)
    ds_list = []
    for ind in indices:
        sub_ds = ds.isel(time=ind)
        # Reassign the time coordinate to ensure it exactly matches the sliced data.
        sub_ds = sub_ds.assign_coords(time=("time", ds.time.values[ind]))
        
        new_attrs = {}
        for key, attr_value in ds.attrs.items():
            try:
                attr_array = np.array(attr_value)
                # Check if it has at least one dimension and its first dimension matches total_times.
                if attr_array.ndim >= 1 and attr_array.shape[0] == total_times:
                    # Slice along the first axis.
                    new_attr = attr_array[ind]
                    if new_attr.ndim == 1:
                        new_attr = new_attr.tolist()
                    new_attrs[key] = new_attr
                else:
                    new_attrs[key] = attr_value
            except Exception:
                # If the attribute cannot be converted to an array, leave it unchanged.
                new_attrs[key] = attr_value
        
        sub_ds.attrs = new_attrs
        ds_list.append(sub_ds)

    return ds_list

def merge_datasets(ds_list):
    """
    Merge a list of xarray datasets along the time dimension.
    """
    merged_ds = xr.concat(ds_list, dim='time')
    # Sort by time in case the datasets are not in order.
    merged_ds = merged_ds.sortby('time')
    return merged_ds

def process_dataset(args):
    """
    Process a single subdataset with additional common parameters.
    
    Parameters:
      args: tuple containing:
        - ds: the xarray subdataset.
        - projection_dim: string specifying projection dimension (e.g., "1d").
        - xyz: additional parameter for reduce_gpu.
        - n_mc: additional parameter for reduce_gpu.
        - vg: additional parameter for reduce_gpu.
    
    Returns:
      The processed xarray dataset.
    """
    ds, projection_dim, xyz, n_mc, vg = args
    #return mms.reduce(ds, projection_dim=projection_dim, xyz=xyz, n_mc=n_mc, vg=vg)
    return mms.reduce_gpu(ds, projection_dim=projection_dim, xyz=xyz, n_mc=n_mc, vg=vg)

def parallel_processing(ds, N, projection_dim, xyz, n_mc, vg):
    """
    Split the dataset into N parts, process each in parallel using reduce_gpu,
    and merge the results back into a single dataset.
    """
    # Split the dataset along the time dimension
    ds_list = split_dataset_by_time(ds, N)
    
    # Build the list of arguments for each process
    args_list = [(sub_ds, projection_dim, xyz, n_mc, vg) for sub_ds in ds_list]
    
    # Process each subdataset in parallel.
    with mp.Pool(processes=N) as pool:
        results = pool.map(process_dataset, args_list)
    
    # Merge the processed subdatasets.
    merged_ds = merge_datasets(results)
    
    return merged_ds


def main():
    t1 = timeit.default_timer()
    r"""Compute 1D reduced ion velocity distribution function."""
    tint = ["2015-12-28T03:57:10", "2015-12-28T03:59:00"]
    mms_id = 2

    # Load data
    # Load magnetic field
    b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, mms_id)

    # Load defatt (spacecraft attitude)
    defatt = mms.load_ancillary("defatt", tint, mms_id)

    # Load ion velocity distribution function
    vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, mms_id)
    vdf_i_err = mms.get_data("pderri_fpi_brst_l2", tint, mms_id)
    vdf_i.data.data[vdf_i.data.data < 1.1 * vdf_i_err.data.data] = 0.0
    
    # Define coordinate system
    n_vec = np.array([0.9580, -0.2708, -0.0938])  # Shock normal
    n_vec /= np.linalg.norm(n_vec)
    b_u = [-1.0948, -2.6270, 1.6478]  # Upstream magnetic field
    t2_vec = np.cross(n_vec, b_u) / np.linalg.norm(
        np.cross(n_vec, b_u)
    )  # Tangent to the shock
    t1_vec = np.cross(t2_vec, n_vec)

    # To time series
    n_t = len(b_dmpa.time.data)
    n_gse = pyrf.ts_vec_xyz(b_dmpa.time.data, np.tile(n_vec[np.newaxis, :], [n_t, 1]))
    t1_gse = pyrf.ts_vec_xyz(b_dmpa.time.data, np.tile(t1_vec[np.newaxis, :], [n_t, 1]))
    t2_gse = pyrf.ts_vec_xyz(b_dmpa.time.data, np.tile(t2_vec[np.newaxis, :], [n_t, 1]))

    # Rotate vectors from GSE to DMPA
    n_dmpa = mms.dsl2gse(n_gse, defatt)
    t1_dmpa = mms.dsl2gse(t1_gse, defatt)
    t2_dmpa = mms.dsl2gse(t2_gse, defatt)

    # Prepare for the projection
    # Create rotation matrices
    nt1t2 = np.transpose(np.stack([n_dmpa.data, t1_dmpa.data, t2_dmpa.data]), [1, 2, 0])
    nt1t2 = pyrf.ts_tensor_xyz(b_dmpa.time.data, nt1t2)

    # Define the velocity grid
    vn_lim = np.array([-800.0, 800.0], dtype=np.float64)
    vg_1d_n = 1e3 * np.linspace(vn_lim[0], vn_lim[1], 100)

    # Reduce the ion VDF to 1D (actual heavy part)
    t2 = timeit.default_timer()
    n_mc = 200
    
    ## CPU serial

    f1dn0 = mms.reduce(vdf_i, projection_dim="1d", xyz=nt1t2, n_mc=n_mc, vg=vg_1d_n)
    # f1dn0.to_netcdf("./data/output/output_reduce.nc")
    
    t3 = timeit.default_timer()
    
    ## GPU parallel

    f1dn = mms.reduce_gpu(vdf_i, projection_dim="1d", xyz=nt1t2, n_mc=n_mc, vg=vg_1d_n)
    #f1dn.to_netcdf("./data/output/output_reduce_gpu.nc")

    t4 = timeit.default_timer()

    # PARALLEL DATASET PROCESSING  with GPU
    N = 4
    merged_dataset = parallel_processing(vdf_i, N, projection_dim="1d", xyz=nt1t2, n_mc=n_mc, vg=vg_1d_n)
    t5 = timeit.default_timer()
    merged_dataset.to_netcdf("./data/output/output_reduce_gpu_parallel.nc")

    print("Time reduce cpu: ",t3-t2)
    print("Time reduce gpu: ",t4-t3)
    print("Time reduce gpu parallel: ",t5-t4)

if __name__ == "__main__":
    main()
