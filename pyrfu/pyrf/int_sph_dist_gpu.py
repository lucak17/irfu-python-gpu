#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
from math import asin, cos, sin, sqrt

import math
import numba
from numba import cuda, float32, int32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import cupy as cp
import numpy as np

#import logging
#logging.getLogger("cupy").setLevel(logging.WARNING)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def int_sph_dist_gpu(vdf, speed, phi, theta, speed_grid, **kwargs):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap.
    speed : numpy.ndarray
        Velocity of the instrument bins,
    phi : numpy.ndarray
        Azimuthal angle of the instrument bins.
    theta : numpy.ndarray
        Elevation angle of the instrument bins.
    speed_grid : numpy.ndarray
        Velocity grid for interpolation.
    **kwargs
        Keyw

    Returns
    -------

    """

    # Coordinates system transformation matrix
    xyz = kwargs.get("xyz", np.eye(3))

    # Number of Monte Carlo iterations and how number of MC points is
    # weighted to data.
    n_mc = kwargs.get("n_mc", 10)
    weight = kwargs.get("weight", None)

    # limit on out-of-plane velocity and azimuthal angle
    v_lim = np.array(kwargs.get("v_lim", [-np.inf, np.inf]), dtype=np.float64)
    a_lim = np.array(kwargs.get("a_lim", [-180.0, 180.0]), dtype=np.float64)
    a_lim = np.deg2rad(a_lim)

    # Projection dimension and base
    projection_base = kwargs.get("projection_base", "pol")
    projection_dim = kwargs.get("projection_dim", "1d")

    speed_edges = kwargs.get("speed_edges", None)
    speed_grid_edges = kwargs.get("speed_grid_edges", None)

    # Azimuthal and elevation angles steps. Assumed to be constant
    # if not provided.
    d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)
    d_phi = kwargs.get("d_phi", d_phi)
    d_theta = np.abs(np.median(np.diff(theta))) * np.ones_like(theta)
    d_theta = kwargs.get("d_theta", d_theta)

    # azimuthal angle of projection plane
    n_az_g = len(phi)
    d_phi_g = 2 * np.pi / n_az_g
    phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, n_az_g) + d_phi_g / 2
    phi_grid = kwargs.get("phi_grid", phi_grid)

    # Overwrite projection dimension if azimuthal angle of projection
    # plane is not provided. Set the azimuthal angle grid width.
    if phi_grid is not None and projection_dim.lower() in ["2d", "3d"]:
        d_phi_grid = np.median(np.diff(phi_grid))
    else:
        projection_dim = "1d"
        d_phi_grid = 1.0

    # Make sure the transformation matrix is orthonormal.
    x_phat = xyz[:, 0] / np.linalg.norm(xyz[:, 0])  # re-normalize
    y_phat = xyz[:, 1] / np.linalg.norm(xyz[:, 1])  # re-normalize

    z_phat = np.cross(x_phat, y_phat)
    z_phat /= np.linalg.norm(z_phat)
    y_phat = np.cross(z_phat, x_phat)

    r_mat = np.transpose(np.stack([x_phat, y_phat, z_phat]), [1, 0])

    if speed_edges is None:
        d_v = np.hstack([np.diff(speed[:2]), np.diff(speed)])
        d_v_m, d_v_p = [np.diff(speed) / 2.0] * 2
    else:
        d_v_m = speed - speed_edges[:-1]
        d_v_p = speed_edges[1:] - speed
        d_v = d_v_m + d_v_p

    # Speed grid bins edges
    if speed_grid_edges is None:
        speed_grid_edges = np.zeros(len(speed_grid) + 1)
        speed_grid_edges[0] = speed_grid[0] - np.diff(speed_grid[:2]) / 2.0
        speed_grid_edges[1:-1] = speed_grid[:-1] + np.diff(speed_grid) / 2.0
        speed_grid_edges[-1] = speed_grid[-1] + np.diff(speed_grid[-2:]) / 2.0
    else:
        speed_grid = speed_grid_edges[:-1] + np.diff(speed_grid_edges) / 2.0

    if projection_base == "pol":
        d_v_grid = np.diff(speed_grid_edges)
    else:
        mean_diff = np.mean(np.diff(speed_grid))
        msg = "For a cartesian grid, all velocity bins must be equal!!"
        assert (np.diff(speed_grid) / mean_diff - 1 < 1e-2).all(), msg

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_sum = n_mc * np.sum(vdf != 0)  # total number of Monte Carlo particles
    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        n_mc_mat = np.ceil(
            n_sum / np.sum(np.log10(vdf + 1)) * np.log10(vdf + 1),
        )
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    n_mc_mat = n_mc_mat.astype(int)

    if projection_base == "cart" and projection_dim == "2d":
        d_a_grid = d_v_grid**2
        f_g = _mc_cart_2d(
            vdf,
            speed,
            phi,
            theta,
            d_v,
            d_v_m,
            d_phi,
            d_theta,
            speed_grid_edges,
            d_a_grid,
            v_lim,
            a_lim,
            n_mc_mat,
            r_mat,
        )
    elif projection_base == "cart" and projection_dim == "3d":
        d_a_grid = d_v_grid**3
        f_g = _mc_cart_3d(
            vdf,
            speed,
            phi,
            theta,
            d_v,
            d_v_m,
            d_phi,
            d_theta,
            speed_grid_edges,
            d_a_grid,
            v_lim,
            a_lim,
            n_mc_mat,
            r_mat,
        )
    else:
        # Area or line element (primed)
        d_a_grid = speed_grid ** (int(projection_dim[0]) - 1) * d_phi_grid * d_v_grid
        d_a_grid = d_a_grid.astype(np.float64)

        if projection_dim == "1d":
            #f_g = _mc_pol_1d_cupy_launcher(
            f_g = _mc_pol_1d_numba_launcher(
                vdf,
                speed,
                phi,
                theta,
                d_v,
                d_v_m,
                d_phi,
                d_theta,
                speed_grid_edges,
                d_a_grid,
                v_lim,
                a_lim,
                n_mc_mat,
                r_mat,
            )
        else:
            raise NotImplementedError(
                "2d projection on polar grid is not ready yet!!",
            )

    if projection_dim == "2d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
        }
    elif projection_dim == "3d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vz": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
            "vz_edges": speed_grid_edges,
        }
    else:
        pst = {"f": f_g, "vx": speed_grid, "vx_edges": speed_grid_edges}

    return pst




def _mc_pol_1d_numba_launcher(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat
    ):

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1

    vdf_dev = numba.cuda.to_device(vdf.flatten())
    v_dev = numba.cuda.to_device(v)
    phi_dev = numba.cuda.to_device(phi)
    theta_dev = numba.cuda.to_device(theta)
    d_v_dev = numba.cuda.to_device(d_v)
    d_v_m_dev = numba.cuda.to_device(d_v_m)
    d_phi_dev = numba.cuda.to_device(d_phi)
    d_theta_dev = numba.cuda.to_device(d_theta)
    vg_edges_dev = numba.cuda.to_device(vg_edges)
    d_a_grid_dev = numba.cuda.to_device(d_a_grid.flatten())
    v_lim_dev = numba.cuda.to_device(v_lim)
    a_lim_dev = numba.cuda.to_device(a_lim)
    n_mc_dev = numba.cuda.to_device(n_mc.flatten())
    r_mat_dev = numba.cuda.to_device(r_mat.flatten())

    f_g_dev = numba.cuda.to_device(np.zeros(n_vg, dtype=np.float32))

    total_bins = n_v * n_ph * n_th
    n_threads = 128
    blocks = ( (total_bins + n_threads - 1) // n_threads, )
    threads = (n_threads,)
    states = create_xoroshiro128p_states(total_bins, seed=5931)
    # Launch the kernel
    _mc_pol_1d_numba_kernel[blocks, threads](vdf_dev, v_dev, phi_dev, theta_dev, d_v_dev, d_v_m_dev, d_phi_dev, d_theta_dev, vg_edges_dev,
                        f_g_dev, d_a_grid_dev, v_lim_dev, a_lim_dev, n_mc_dev, r_mat_dev, n_v, n_ph, n_th, n_vg, states)
    
    f_g_result = f_g_dev.copy_to_host()

    return f_g_result







@numba.cuda.jit(cache=True, fastmath=True, debug=False)
def _mc_pol_1d_numba_kernel(vdf, v, phi, theta, d_v, d_v_m, d_phi, d_theta, vg_edges, f_g, d_a_grid,
                            v_lim, a_lim, n_mc, r_mat, n_v, n_ph, n_th, n_vg, states):
    """
    Monte Carlo kernel
    
    Parameters:
      vdf      : 1D array of doubles, instrument VDF, flattened shape = n_v*n_ph*n_th.
      v        : 1D array of doubles, instrument speed centers, length n_v.
      phi      : 1D array of doubles, azimuth centers, length n_ph.
      theta    : 1D array of doubles, elevation centers, length n_th.
      d_v      : 1D array of doubles, speed bin widths, length n_v.
      d_v_m    : 1D array of doubles, “minus” speed offset (only d_v_m[0] is used).
      d_phi    : 1D array of doubles, azimuth bin widths, length n_ph.
      d_theta  : 1D array of doubles, elevation bin widths, length n_th.
      vg_edges : 1D array of doubles, grid velocity edges, length n_vg+1.
      f_g      : 1D array of float32, output grid; length n_vg.
      d_a_grid : 1D array of doubles, projection grid bin widths, length n_vg.
      v_lim    : 1D array of doubles, velocity limits [v_min, v_max].
      a_lim    : 1D array of doubles, angular limits [a_min, a_max].
      n_mc     : 1D array of ints, number of Monte-Carlo samples per instrument bin (flattened), length = n_v*n_ph*n_th.
      r_mat    : 1D array of doubles, 3x3 transformation matrix (row-major), length 9.
      n_v, n_ph, n_th : ints, dimensions of the instrument VDF.
      n_vg     : int, number of grid bins.
      states   : random state array created with create_xoroshiro128p_states.
    """
    # Compute the linear instrument bin index.
    idx = numba.cuda.blockIdx.x * numba.cuda.blockDim.x + numba.cuda.threadIdx.x
    total_bins = n_v * n_ph * n_th
    if idx >= total_bins:
        return

    # Compute instrument bin indices: i, j, k.
    i = idx // (n_ph * n_th)
    remainder = idx % (n_ph * n_th)
    j = remainder // n_th
    k = remainder % n_th

    # Get number of Monte Carlo samples for this instrument bin.
    n_mc_ijk = n_mc[idx]
    if n_mc_ijk <= 0:
        return

    # Use the thread's index to access its random state.
    # (Assuming states has been allocated with at least total_bins entries.)
    # Each call updates the state.
    # Note: xoroshiro128p_uniform_float64 returns a float in [0, 1).
    rand1 = xoroshiro128p_uniform_float64(states, idx)
    rand2 = xoroshiro128p_uniform_float64(states, idx)
    rand3 = xoroshiro128p_uniform_float64(states, idx)
    
    # Load instrument bin parameters.
    vi    = v[i]
    phij  = phi[j]
    thetak= theta[k]
    dv_i  = d_v[i]
    dv_m0 = d_v_m[0]
    dphij = d_phi[j]
    dthetak = d_theta[k]

    # Compute dtau = v[i]^2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
    dtau = vi * vi * math.cos(thetak) * dv_i * dphij * dthetak
    c_ijk = dtau / float(n_mc_ijk)
    f_ijk = vdf[idx]

    # Loop over Monte Carlo samples for this instrument bin.
    for m in range(n_mc_ijk):
        # Generate three pseudo-random numbers for this iteration.
        r1 = xoroshiro128p_uniform_float64(states, idx)
        r2 = xoroshiro128p_uniform_float64(states, idx)
        r3 = xoroshiro128p_uniform_float64(states, idx)
        
        d_v_mc   = -r1 * dv_i - dv_m0
        d_phi_mc = (r2 - 0.5) * dphij
        d_the_mc = (r3 - 0.5) * dthetak

        # Compute perturbed values.
        v_mc    = vi + d_v_mc
        phi_mc  = phij + d_phi_mc
        theta_mc= thetak + d_the_mc

        # Convert from spherical to cartesian coordinates.
        v_x = v_mc * math.cos(theta_mc) * math.cos(phi_mc)
        v_y = v_mc * math.cos(theta_mc) * math.sin(phi_mc)
        v_z = v_mc * math.sin(theta_mc)

        # Transform the velocity vector using r_mat (row-major order).
        v_x_p = r_mat[0]*v_x + r_mat[3]*v_y + r_mat[6]*v_z
        v_y_p = r_mat[1]*v_x + r_mat[4]*v_y + r_mat[7]*v_z
        v_z_p = r_mat[2]*v_x + r_mat[5]*v_y + r_mat[8]*v_z

        # Compute a new v_z_p as sqrt(v_y_p^2 + v_z_p^2)
        v_z_p = math.sqrt(v_y_p*v_y_p + v_z_p*v_z_p)

        # Compute alpha = asin(v_z_p / v_mc), ensuring the ratio is not >1.
        ratio = v_z_p / v_mc
        if ratio > 1.0:
            ratio = 1.0
        alpha = math.asin(ratio)

        # Check if the perturbed sample falls within the velocity and angular limits.
        if (v_z_p >= v_lim[0] and v_z_p < v_lim[1] and
            alpha >= a_lim[0] and alpha < a_lim[1]):
            # Find the grid bin index by scanning vg_edges.
            i_vxg = 0
            for e in range(n_vg):
                i_vxg = e
                if v_x_p < vg_edges[e]:
                    break
            if i_vxg < n_vg:
                add_value = f_ijk * c_ijk / d_a_grid[i_vxg]
                add_val = float(add_value)  # Convert to float32 later via atomic add.
                # Use atomic add on the output grid.
                numba.cuda.atomic.add(f_g, i_vxg, add_val)
    # End of kernel.








# NOT use this kernel! it gives incorrect results
def _mc_pol_1d_cupy_launcher(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat
    ):

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    
    vdf_dev = cp.asarray(vdf.flatten())
    v_dev = cp.asarray(v)
    phi_dev = cp.asarray(phi)
    theta_dev = cp.asarray(theta)
    d_v_dev = cp.asarray(d_v)
    d_v_m_dev = cp.asarray(d_v_m)
    d_phi_dev = cp.asarray(d_phi)
    d_theta_dev = cp.asarray(d_theta)
    vg_edges_dev = cp.asarray(vg_edges)
    d_a_grid_dev = cp.asarray(d_a_grid.flatten())
    v_lim_dev = cp.asarray(v_lim)
    a_lim_dev = cp.asarray(a_lim)
    n_mc_dev = cp.asarray(n_mc.flatten())
    r_mat_dev = cp.asarray(r_mat.flatten())

    f_g_dev = cp.zeros(n_vg, dtype=cp.float32)

    n_threads = 128
    blocks = ( (n_v*n_ph*n_th  + n_threads - 1) // n_threads, )
    threads = (n_threads,)
    cp.cuda.Stream.null.synchronize()
    # Call the kernel:
    mc_cart_1d_kernel_cupy(
        blocks, threads,
        (vdf_dev, v_dev, phi_dev, theta_dev, d_v_dev, d_v_m_dev, d_phi_dev, d_theta_dev, vg_edges_dev,
        f_g_dev, d_a_grid_dev, v_lim_dev, a_lim_dev, n_mc_dev, r_mat_dev,
        n_v, n_ph, n_th, n_vg ) )
        
    f_g_result = f_g_dev.get()

    return f_g_result



# NOT use this kernel! it gives incorrect results
kernel_code = r'''
#include <curand_kernel.h>
extern "C" __global__
void mc_cart_1d_kernel(
    const double*  vdf,    // instrument VDF, shape: n_v * n_ph * n_th (flattened)
    const double*  v,      // instrument speed centers, length: n_v
    const double*  phi,    // azimuth centers, length: n_ph
    const double*  theta,  // elevation centers, length: n_th
    const double*  d_v,    // speed bin widths, length: n_v
    const double*  d_v_m,  // "minus" speed offset, only d_v_m[0] is used
    const double*  d_phi,  // azimuth bin widths, length: n_ph
    const double*  d_theta,// elevation bin widths, length: n_th
    const double*  vg_edges, // grid velocity edges, length: n_vg+1 (sorted)
    float*  f_g,          // output interpolated grid, shape: (n_vg
    const double*  d_a_grid,             // projection grid bin width
    const double*  v_lim,  // velocity limits, length 2: [v_min, v_max]
    const double*  a_lim,  // angular limits, length 2: [a_min, a_max]
    const int*  n_mc,      // number of Monte-Carlo samples per instrument bin, shape: n_v * n_ph * n_th (flattened)
    const double*  r_mat,  // 3x3 frame transformation matrix (row-major order)
    const int n_v,
    const int n_ph,
    const int n_th,
    const int n_vg   // number of grid bins
)
{
    // Compute instrument bin indices
    //const int i = blockIdx.x * blockDim.x + threadIdx.x;
    //const int j = blockIdx.y * blockDim.y + threadIdx.y;
    //const int k = blockIdx.z * blockDim.z + threadIdx.z;
    // if(i >= n_v || j >= n_ph || k >= n_th) return;
    
    // Linear index for the instrument bin
    // const int idx = i * (n_ph * n_th) + j * n_th + k;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx / (n_ph * n_th);
    const int remainder = idx % (n_ph * n_th);
    const int j = remainder / n_th;
    const int k = remainder % n_th;
    if(idx >= n_v * n_ph * n_th) return;

    const int n_mc_ijk = n_mc[idx];
    
    // If no particles, skip
    //if(abs(vdf[idx]) <  1e-20)return;

    curandState localState1;
    curand_init(100807677, idx, 0, &localState1);
    curandState localState2;
    curand_init(100807677, idx, 0, &localState2);
    curandState localState3;
    curand_init(100807677, idx, 0, &localState3);
    

    // Load instrument bin values
    const double vi    = v[i];
    const double phij  = phi[j];
    const double thetak= theta[k];
    const double dv_i  = d_v[i];
    const double dv_m0 = d_v_m[0];
    const double dphij = d_phi[j];
    const double dthetak = d_theta[k];
    
    // Compute dtau = v[i]^2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
    double dtau = vi * vi * cos(thetak) * dv_i * dphij * dthetak;
    double c_ijk = dtau / ((double)n_mc_ijk);
    double f_ijk = vdf[idx];
    // Loop over Monte-Carlo samples for this instrument bin
    for (int m = 0; m < n_mc_ijk; m++) {
        // Generate three pseudo-random numbers in [0,1)
        // A simple LCG is used here; combine the instrument bin index and the Monte-Carlo index.
        const double rand1 = curand_uniform(&localState1);
        const double rand2 = curand_uniform(&localState2);
        const double rand3 = curand_uniform(&localState3);

        double d_v_mc   = -rand1 * dv_i - dv_m0;
        double d_phi_mc = (rand2 - 0.5) * dphij;
        double d_the_mc = (rand3 - 0.5) * dthetak;

        // Compute perturbed values
        double v_mc    = vi + d_v_mc;
        double phi_mc  = phij + d_phi_mc;
        double theta_mc= thetak + d_the_mc;
        
        // Convert from spherical to cartesian coordinates
        double v_x = v_mc * cos(theta_mc) * cos(phi_mc);
        double v_y = v_mc * cos(theta_mc) * sin(phi_mc);
        double v_z = v_mc * sin(theta_mc);
        
        // Transform the velocity vector using r_mat (assumed row-major)
        double v_x_p = r_mat[0] * v_x + r_mat[3] * v_y + r_mat[6] * v_z;
        double v_y_p = r_mat[1] * v_x + r_mat[4] * v_y + r_mat[7] * v_z;
        double v_z_p = r_mat[2] * v_x + r_mat[5] * v_y + r_mat[8] * v_z;
        
        v_z_p = sqrt(v_y_p * v_y_p + v_z_p * v_z_p);
    
        double alpha = asin(v_z_p / v_mc);
        
        if ((v_z_p >= v_lim[0]) && (v_z_p < v_lim[1]) && (alpha >= a_lim[0]) && (alpha < a_lim[1])) {
            int i_vxg = 0;
            for (int e = 0; e < n_vg; e++) {
                i_vxg = e;
                if(v_x_p < vg_edges[e]){
                    break;
                }
            }           
            // Use atomic addition (CUDA supports atomicAdd on double on supported architectures)
            if (i_vxg < n_vg){
                double add_value = f_ijk * c_ijk / d_a_grid[i_vxg]; 
                float add_val = (float)add_value;
                atomicAdd(&f_g[i_vxg] , add_val);
                __syncthreads();
            }
        }
    }

}
'''

    # Compile the kernel
mc_cart_1d_kernel_cupy = cp.RawKernel(kernel_code, 'mc_cart_1d_kernel', options=(
        '--std=c++17',
        '-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',))



@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_3d(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : numpy.ndarray
        3D skymap particle velocity distribution function.
    v : numpy.ndarray
        1D array of instrument speed bins centers.
    phi : numpy.ndarray
        1D array of instrument azimuthal angles bins centers.
    theta : numpy.ndarray
        1D array of instrument elevation angles bins centers.
    d_v : numpy.ndarray
        1D array of instrument speed bins widths.
    d_v_m : numpy.ndarray
        1D array of minus velocity from bins centers.
    d_phi : numpy.ndarray
        1D array of instrument azimuthal angles bins widths.
    d_theta : numpy.ndarray
        1D array of instrument elevation angles bins widths.
    vg_egdes : double
        Bin centers of the velocity of the projection grid.
    d_a_grid : double
        Bin centers of the azimuthal angle of the projection in radians in
        the span [0,2*pi]. If this input is given, the projection will be 2D.
        If it is omitted, the projection will be 1D.
    v_lim : double
        Limits on the out-of-plane velocity interval in 2D and "transverse"
        velocity in 1D.
    a_lim : double
        Angular limit in degrees, can be combined with v_lim.
    n_mc : double
        Number of Monte-Carlo particle for the corresponding instrument bins.
    r_mat : double
        Frame transformation matrix.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg, n_vg))

    for i in numba.prange(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v[i] ** 2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for _ in range(n_mc_ijk):
                    d_v_mc = -random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (random.random() - 0.5) * d_phi[j]
                    d_the_mc = (random.random() - 0.5) * d_theta[k]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = theta[k] + d_the_mc

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z
                    # velocity within [-dVm, +dVp]

                    alpha = asin(v_z_p / v_mc)

                    use_point = v_z_p >= v_lim[0] * v_z_p < v_lim[1]
                    use_point = use_point * alpha >= a_lim[0] * alpha < a_lim[1]

                    i_vxg = np.searchsorted(vg_edges[:-2], v_x_p)
                    i_vyg = np.searchsorted(vg_edges[:-2], v_y_p)
                    i_vzg = np.searchsorted(vg_edges[:-2], v_z_p)

                    if use_point:
                        f_g[i_vxg, i_vyg, i_vzg] += f_ijk * c_ijk / d_a_grid

    return f_g


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_2d(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : double
        3D skymap particle velocity distribution function.
    v : double
        1D array of instrument speed bins centers.
    phi : double
        1D array of instrument azimuthal angles bins centers.
    theta : double
        1D array of instrument elevation angles bins centers.
    d_v : double
        1D array of instrument speed bins widths.
    d_v_m : double
        1D array of minus velocity from bins centers.
    d_phi : double
        1D array of instrument azimuthal angles bins widths.
    d_theta : double
        1D array of instrument elevation angles bins widths.
    vg_egdes : double
        Bin centers of the velocity of the projection grid.
    d_a_grid : double
        Bin centers of the azimuthal angle of the projection in radians in
        the span [0,2*pi]. If this input is given, the projection will be 2D.
        If it is omitted, the projection will be 1D.
    v_lim : double
        Limits on the out-of-plane velocity interval in 2D and "transverse"
        velocity
        in 1D.
    a_lim : double
        Angular limit in degrees, can be combined with v_lim.
    n_mc : double
        Number of Monte-Carlo particle for the corresponding instrument bins.
    r_mat : double
        Frame transformation matrix.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg))

    for i in range(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v[i] ** 2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for _ in range(n_mc_ijk):
                    d_v_mc = -random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (random.random() - 0.5) * d_phi[j]
                    d_the_mc = (random.random() - 0.5) * d_theta[k]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = theta[k] + d_the_mc

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z
                    # velocity within [-dVm, +dVp]

                    alpha = asin(v_z_p / v_mc)

                    use_point = v_z_p >= v_lim[0] * v_z_p < v_lim[1]
                    use_point = use_point * alpha >= a_lim[0] * alpha < a_lim[1]

                    i_vxg = np.searchsorted(vg_edges, v_x_p)
                    i_vyg = np.searchsorted(vg_edges, v_y_p)

                    if use_point:
                        f_g[i_vxg, i_vyg] += f_ijk * c_ijk / d_a_grid

    return f_g
