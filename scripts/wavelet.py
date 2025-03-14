#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from pyrfu import mms, pyrf
import timeit
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__license__ = "MIT"

# Initialize MMS client
mms.db_init(default="local", local="./data/")


def main():
    r"""Compute wavelet transform."""
    
    # Define time interval and spacecraft index
    tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
    mms_id = 1

    # Load data
    # Load magnetic field FGM
    b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint, mms_id)

    # Load electric field
    e_xyz = mms.get_data("e_gse_edp_brst_l2", tint, mms_id)

    # Some pre-processing
    # Rotate E and B into field-aligned coordinates
    e_fac = pyrf.convert_fac(e_xyz, b_xyz, [1, 0, 0])

    # Bandpass filter E and B waveforms
    fmin, fmax = [0.5, 1000]  # Hz

    # Compute wavelet transform (this is the heavy part)
    nf = 100

    # Save using NetCDF4 to avoid int32 restriction
    t1 = timeit.default_timer()
    # e_cwt0 = pyrf.wavelet(e_fac, f=[fmin, fmax], n_freqs=nf)
    t2 = timeit.default_timer()
    #e_cwt0.to_netcdf("./data/output/output_wavelet_orig.nc")
    

    e_cwt = pyrf.wavelet_gpu(e_fac, f=[fmin, fmax], n_freqs=nf)
    t3 = timeit.default_timer()    
    
    if "time" in e_cwt:
        # Convert datetime64 to seconds since 1970
        if np.issubdtype(e_cwt["time"].dtype, np.datetime64):
            e_cwt["time"] = ("time", (e_cwt["time"].values - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))

        # Explicitly set time encoding to float64 to silence the warning
        e_cwt["time"].encoding["dtype"] = np.float64
        e_cwt["time"].encoding["units"] = "seconds since 1970-01-01"

    # Save using NetCDF4 to avoid int32 restriction
    #e_cwt.to_netcdf("./data/output/output_wavelet_gpu.nc")
    

    print("Time wavelet cpu: ",t2-t1)
    print("Time wavelet gpu: ",t3-t2)

if __name__ == "__main__":
    main()
