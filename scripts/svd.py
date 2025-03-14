#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from pyrfu import mms, pyrf
import timeit

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__license__ = "MIT"

# Initialize MMS client
mms.db_init(default="local", local="./data/")


def main():    
    r"""Perform polarization analysis."""
    t1 = timeit.default_timer()
    # Define time interval and spacecraft index
    tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.00"]
    tint_long = pyrf.extend_tint(tint, [-100, 100])
    mms_id = 1

    # Load data
    r_xyz = mms.get_data("r_gse_mec_srvy_l2", tint_long, mms_id)
    b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint, mms_id)
    e_xyz = mms.get_data("e_gse_edp_brst_l2", tint, mms_id)
    b_scm = mms.get_data("b_gse_scm_brst_l2", tint, mms_id)

    t2 = timeit.default_timer()
    # Perform the polarization analysis with CPU (this is the heavy part)
    polarization0 = pyrf.ebsp( e_xyz, b_scm, b_xyz, b_xyz, r_xyz, freq_int=[10, 4000], polarization=True, fac=True, )
    t3 = timeit.default_timer()
    
    # Polarization analysis on GPU
    polarization = pyrf.ebsp_gpu(e_xyz, b_scm, b_xyz, b_xyz, r_xyz, freq_int=[10, 4000], polarization=True, fac=True, )
    
    t4 = timeit.default_timer()   
    #polarization.to_netcdf("../data/output_polarization.nc")
    # Fix time encoding issue
    #polarization["t"].encoding["dtype"] = "int32"  # Force int32 time encoding
    #polarization["t"].encoding["units"] = "seconds since 1970-01-01"
    
    if "t" in polarization:
        polarization["t"].encoding.pop("dtype", None)  # Remove dtype encoding
        polarization["t"].encoding["units"] = "seconds since 1970-01-01"

    # Save to NetCDF with correct encoding
    #polarization.to_netcdf("../data/output_polarization_gpu.nc")

    print("Time svd cpu: ",t3-t2)
    print("Time svd gpu: ",t4-t3)

if __name__ == "__main__":
    main()
