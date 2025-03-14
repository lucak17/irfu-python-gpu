#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
from typing import Dict, Optional, Union

# 3rd party imports
import numba
import numpy as np
import cupy as cp
import xarray as xr
from numpy.typing import NDArray
from scipy import fft
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from .calc_fs import calc_fs

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _ww(
    s_ww: NDArray[cp.complex128],
    scales_mat: NDArray[cp.float64],
    sigma: float,
    frequencies_mat: NDArray[cp.float64],
    f_nyq: float ) -> NDArray[cp.complex128]:
    w_w: NDArray[cp.complex128] = s_ww * cp.exp( -sigma * sigma * ((scales_mat * frequencies_mat - f_nyq) ** 2) / 2, )
    return w_w

def _power_r(
    power: NDArray[cp.complex128], new_freq_mat: NDArray[cp.float64]
) -> NDArray[cp.float64]:
    power2: NDArray[cp.float64] = cp.absolute( (2 * cp.pi) * cp.conj(power) * power / new_freq_mat )
    return power2

def _power_c(power, new_freq_mat):
    power2 = ( cp.sqrt(cp.absolute((2 * cp.pi) / new_freq_mat)) * power )
    return power2




def wavelet_gpu(
    inp: DataArray,
    f_s: Optional[float] = None,
    f: Optional[list[float]] = None,
    n_freqs: Optional[int] = None,
    linear: Optional[Union[float, bool]] = None,
    wavelet_width: Optional[float] = None,
    cut_edge: Optional[bool] = True,
    return_power: Optional[bool] = True,
) -> Union[DataArray, Dataset]:
    """Computes wavelet spectrogram based on fast FFT algorithm.
    Parameters
    ----------
    inp : DataArray
        Input quantity.
    f_s : float, Optional
        Sampling frequency of the input time series.
    f : list, Optional
        Vector [f_min f_max], calculate spectra between frequencies
        f_min and f_max.
    n_freqs : int, Optional
        Number of frequency bins.
    linear : float or bool, Optional
        Linear spacing between frequencies of df.
    wavelet_width : float, Optional
        Width of the Morlet wavelet. Default 5.36.
    cut_edge : bool, Optional
        Set to True to set points affected by edge effects to NaN,
        False to keep edge affect points. Default True
    return_power : bool, Optional
        Set to True to return the power, False for complex wavelet
        transform. Default True.

    Returns
    -------
    DataArray or Dataset
        Wavelet transform of the input.


    Raises
    ------
    TypeError
        If linear keyword argument is not bool or float.
    ValueError
        If input is not 1D or 2D.

    """

    # Check input
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    if f_s is None:
        f_s = calc_fs(inp)

    if n_freqs is None:
        n_freqs = 200

    if wavelet_width is None:
        wavelet_width = 5.36

    if linear is not None:
        if isinstance(linear, float):
            delta_f: float = linear
            linear_df: bool = True
        elif isinstance(linear, bool) and linear:
            delta_f = 100.0
            linear_df = True
            logging.warning("Unknown input for linear delta_f set to 100")
        else:
            raise TypeError("linear keyword argument must be bool or float")
    else:
        delta_f = 100.0
        linear_df = False

    # Nyquist frequency and wavelet width
    f_nyq: float = f_s / 2
    sigma: float = wavelet_width / f_nyq

    # Frequency range
    if f is None:
        f_min: float = f_nyq / 10**2
        f_max: float = f_nyq / 10**-2
    else:
        f_min, f_max = sorted(f)

    if linear_df:
        scale_number: int = int(np.floor(f_nyq / delta_f))

        # Scales range
        scale_min: float = delta_f
        scale_max: float = scale_number * delta_f
        scales: NDArray[np.float64] = f_nyq / (
            np.linspace(scale_max, scale_min, scale_number, dtype=np.float64)
        )
    else:
        scale_number = n_freqs
        scale_min = np.log10(f_nyq / f_max)
        scale_max = np.log10(f_nyq / f_min)
        scales = np.logspace(scale_min, scale_max, scale_number, dtype=np.float64)

    # Unpack time and data.
    # Remove the last sample if the total number of samples is odd.
    if len(inp.time.data) % 2:
        time: NDArray[np.datetime64] = inp.time.data[:-1]
        data: NDArray[np.float64] = inp.data[:-1, ...].astype(np.float64)
    else:
        time = inp.time.data
        data = inp.data.astype(np.float64)

    # Preallocate power2
    if return_power:
        power2: NDArray[Union[np.float64, np.complex128]] = np.zeros(
            (len(time), n_freqs), dtype=np.float64
        )
    else:
        power2 = np.zeros((len(time), n_freqs), dtype=np.complex128)

    # Check for NaNs
    scales[np.isnan(scales)] = 0.0

    # Find the frequencies for an FFT of all data
    freq: NDArray[np.float64] = (
        f_nyq * np.arange(1, 1 + len(data) / 2) / (len(data) / 2)
    )

    # The frequencies corresponding to FFT
    freqs_fft: NDArray[np.float64] = np.hstack([0, freq, -np.flip(freq[:-1])])
    _, freqs_fft_mat = np.meshgrid(scales, freqs_fft, sparse=True)

    # Get the correct frequencies for the wavelet transform
    freqs_cwt: NDArray[np.float64] = f_nyq / scales
    freqs_cwt_mat, _ = np.meshgrid(freqs_cwt, freqs_fft, sparse=True)

    if data.ndim in [1, 2]:
        out_dict: Dict[str, object] = {}
    else:
        raise ValueError("Input data must be 1D or 2D")

    # if scalar add virtual axis
    if len(inp.shape) == 1:
        data = data[:, np.newaxis]


    # GPU arrays
    length_data = len(data[:, 0])
    freqs_fft_mat_dev = cp.asarray(freqs_fft_mat)
    tile = np.tile(freqs_cwt_mat, (length_data, 1))
    newfreqs_cwt_mat_dev = cp.asarray(tile)

    scales_dev = cp.asarray(scales)
    censure_dev = cp.floor(2 * scales_dev).astype(cp.int32)
    rows, cols = cp.ogrid[:length_data, :scale_number]
    mask_lower = rows < censure_dev  
    # Mask for rows in the upper part: rows greater than or equal to (length_data - censure) for each column.
    mask_upper = rows >= (length_data - censure_dev)
    data_all_dev = cp.asarray(data)
    for i in range(data.shape[1]):
                
        data_dev = data_all_dev[:,i]
        # Forward FFT on the GPU (cuFFT is used under the hood)
        s_w_dev : NDArray[cp.complex128] = cp.fft.fft(data_dev)
        scales_mat_dev, s_w_mat_dev = cp.meshgrid(scales_dev, s_w_dev, sparse=True)
        # Calculate the FFT of the wavelet transform
        w_w_dev  = _ww( s_w_mat_dev, scales_mat_dev, sigma, freqs_fft_mat_dev, f_nyq )
        # Backward FFT
        power_dev : NDArray[cp.complex128] = cp.fft.ifft(w_w_dev, axis=0)
        # Calculate the power spectrum    
        if return_power:
            power2_dev = _power_r(power_dev, newfreqs_cwt_mat_dev)
        else:
            power2_dev = _power_c(power_dev, newfreqs_cwt_mat_dev)
        if cut_edge:
            power2_dev[mask_lower] = cp.nan
            power2_dev[mask_upper] = cp.nan

        power2 = power2_dev.get()

        if len(inp.shape) == 2:
            # Construct xarray.DataArray here
            out_dict[str(inp.comp.data[i])] = (
                ["time", "frequency"],
                np.fliplr(power2),
            )

    if len(inp.shape) == 1:
        out: Union[DataArray, Dataset] = xr.DataArray(
            np.fliplr(power2),
            coords=[time, np.flip(freqs_cwt)],
            dims=["time", "frequency"],
        )
    else:
        out = xr.Dataset(
            out_dict,
            coords={"time": time, "frequency": np.flip(freqs_cwt)},
        )

    return out
