""" PSF generation and precise localization of single molecules in 3D"""

#%%

# core
import numpy as np
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches

# plotly
import dash
from dash import dcc, html, Input, Output, State
from dash.dash_table import DataTable
# from dash_extensions.snippets import send_data_frame
# from dash_extensions import Download
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from io import BytesIO


# jax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.numpy.linalg import solve,norm
from jax.config import config
import jax.scipy as jsp
config.update("jax_enable_x64", False)

# skimage
from skimage.feature import peak_local_max

from sklearn.linear_model import LinearRegression

from functools import partial
import aotools.functions.zernike as ao
import copy

# scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import mat73

# import _bspline



#%% INDEX OF CONTENT

# CAMERA AND OBJECTIVE CLASSES


# PSF CLASS

    # INTERPOLATION
    #   - f
    #   - f_single
    #   - f_biplane

    # FIELD CALCULATION
    #   - calc_fresnel_coefs
    #   - calc_BFP_fields
    #   - calc_czt2_coefs
    #   - czt2
    #   - calc_molecule_image

    # PUPIL PHASE
    #   - calc_zernike
    #   - calc_aberrations    
    #   - calc_apodization
    #   - calc_defocus
    #   - calc_SA
    #   - RI_mismatch_correction
    #   - calc_CRLB

    # PLOT
    #   - show_slice_xy
    #   - show_projection
    #   - show_pupil
    #   - show_CRLB


# OTHER FUNCTIONS

#   PSF SUPPORT FUNCTIONS
#   - Fisher
#   - add_CRLB
#   - combine_PSFs
#   - embed
#   - create_coord
#   - load_PSF
#   - binning

#   PRELOC FUNCTIONS
#   - prepare_images_for_fitting
#   - draw_peaks
#   - crop_images
#   - filter_biplane_double_peaks
#   - preloc_setup
#   - preloc_and_crop
#   - add_preloc_positions

#   FIT FUNCTIONS
#   - estimate_v0
#   - perform_fit
#   - lm_poisson
#   - fit_step

#   - show_prelocs
#   - show_results
#   - filt_results



#%%


# CAMERA AND OBJECTIVE CLASSES



class Camera:
    def __init__(self, pixsize = 6500, QE = 0.75, baseline = 100, amp = 0.24, EM_gain = 1, readnoise = 1):
        '''
        # Formula to get Photon number from counts:
        Img[photons] = (Img[counts] - baseline[counts]) * amp[electrons/count] / QE[electrons/photon] 

        Be Aware: amp is named Conversion Factor by Hamamatsu and Sensitivity by Andor.
        it is the number of electrons that you need to make 1 count.
        ''' 
        self.pixsize = pixsize                  # pixel size in nm
        self.QE = QE                            # conversion factor from electrons to photons
        self.baseline = baseline                # dark offset, in counts
        self.EM_gain = EM_gain                  # electrons multiplicative factor. for non EMCCDs is 1
        self.amp = amp / EM_gain                # conversion factor from counts to electrons, normalized by EM_gain
        self.readnoise = readnoise / EM_gain    # in electrons rms, normalized by EM_gain




class Objective:
    def __init__(self, M = 60, NA = 1.49, RI = 1.518, focal_len = 3e6, Ts_coefs = [1], Tp_coefs = [1]):
        self.M = M
        self.NA = NA
        self.RI = RI
        self.focal_len = focal_len      # nm
        self.Ts_coefs = Ts_coefs
        self.Tp_coefs = Tp_coefs



# PSF CLASS

class psf:
    def __init__(self, 
                 NA = 1.49, 
                 wavelength = 670,          # nm
                 z_defocus = 400,           # nm
                 RI = [1.33, 1.518],        # sample, immersion medium
                 Nk = 128, 
                 Nx = 17, 
                 ux = 108.3,                # nm
                 uz = 5,                    # nm
                 focal_len = 3e6,           # nm
                 z_min = 0,                 # nm
                 z_max = 300,               # nm
                 Z_modes = [5, 6, 7, 8, 11],# noll indices
                 Z_magn = [0, 0, 0, 0, 0],  # noll amplitudes, in rad
                 Ts_coefs = [1],
                 Tp_coefs = [1],
                 T_fit_coefs = [1],
                 d2 = 0.,                    # nm
                 os = 1,
                 sigma = 100,               # nm
                 interp_method = 'lanczos5',
                 correct_RI_mismatch = 'no'):
        
        '''initialize psf base parameters'''
        self.NA = NA                                # NA of objective
        self.RI = RI                                # list of RIs: [sample, (intermediate layer, when present), immersion medium]
        self.wavelength = wavelength * 1e-9         # vacuum wavelength of peak emission in m
        self.focal_len = focal_len * 1e-9           # objective focal lenght in m
        self.Nk = Nk                                # grid size in k space. Must be a power of 2 for computational efficiency.

        '''initialize psf size parameters'''
        self.z_min = z_min * 1e-9                   # PSF start z value. Distance from coverslip in m
        self.z_max = z_max * 1e-9                   # PSF end z value. Distance from coverlip in m
        self.uz = uz * 1e-9                         # PSF z step size in m
        self.Nx = Nx                                # PSF pixel size in real space. Should be 4 pixels larger than the images to fit for better interpolation.
        self.ux = ux * 1e-9                         # PSF effective pixel size in m: sensor pixel size / objective magnification
        self.os = os                                # oversampling. integer.

        '''initialize psf properties'''
        self.d2 = d2 * 1e-9                         # intermediate layer thickness in m
        self.z_defocus = z_defocus * 1e-9           # distance between the coverslip and the nominal objective focal plane. in m
        self.sigma_blurring = sigma * os / ux       # gaussian sigma to blur the PSF. in units of camera effective pixel size.
        
        self.Z_modes = jnp.array(Z_modes)           # aberrations Noll index
        self.Z_magn = jnp.array(Z_magn)             # aberrations amplitude (radians).
        self.Ts_coefs = jnp.array(Ts_coefs)         # objective s apodization coefficients
        self.Tp_coefs = jnp.array(Tp_coefs)         # objective p apodization coefficients
        self.T_fit_coefs = jnp.array(T_fit_coefs)   # field transmission
        self.interp_method = interp_method          # interpolation method: tricubic or lanczos5

        ''' PSF functions (keep this order) '''
        self.parameters()           # computes parameters needed in the following functions
        self.apodization()          # computes the objective apodization of BFP electric fields
        self.zernikes()             # creates the Zernike basis set: modeset and modeset_fftshift
        self.fresnel_coefs()        # computes Fresnel coefficients
        self.czt2_coefs()           # computes parameters needed in the 2d chirp z-transform
        self.gaussian_kernel()      # create the gaussian kernel for the PSF blurring

        ''' Electric fields in the BFP. [E_BFP shape: (6, len(z_vec), Nk, Nk)]'''
        self.Exx, self.Eyx = self.BFP_fields([jnp.pi/2,0])             # emitting dipole oriented along x
        self.Exy, self.Eyy = self.BFP_fields([jnp.pi/2, jnp.pi/2])     # emitting dipole oriented along y
        self.Exz, self.Eyz = self.BFP_fields([0,0])                    # emitting dipole oriented along z
        self.E_BFP = jnp.array([self.Exx, self.Eyx, self.Exy, self.Eyy, self.Exz, self.Eyz])[..., jnp.newaxis].transpose((0, 3, 1, 2)) # newaxis for z dim

        ''' PUPIL contributions: aberrations and objective defocus '''
        self.pupil_aberrations = self.aberrations()  # Pupil phase given by aberrations (Defocus Noll idx = 4 not included). fft shifted
        self.obj_defocus,_ = self.calc_defocus( - self.z_defocus, self.RI[2]) # pupil phase given by objective nominal defocus (in immersion medium). fft shifted
        # Phase correction of defocus due to RI mismatch. fft shifted.
        self.calc_SA( - self.z_defocus)  # This term is implicitly inlcuded in the Axelrod model. Here we calculate it explicitly for visualization, RI correction or zernike decomposition.
        self.RI_mismatch_correction(correct_RI_mismatch)  # apply spherical aberration correction if desired: tot_defocus = defocus + (SA or SA_pure or 0). fft shifted

        '''Generate the PSF'''
        self.exp_jphase = jnp.exp(1j*jnp.real(self.tot_defocus + self.pupil_aberrations)) * self.apodization # pupil phase that creates the PSF. fft shifted.
        self.data, self.data_fields = self.generate_psf(self.z_vec, self.exp_jphase) #calculate the 3D PSF with blurring




    # INTERPOLATION FUNCTION f
    #   - f
    #   - f_single
    #   - f_biplane
    #   - generate_image
    
    #@partial(jax.jit, static_argnums=(0,))
    def f(self, v):  # f = inteporlation function
        if isinstance(self.z_defocus, list):
            return self.f_biplane(v)
        else:
            return self.f_single(v)

    
    @partial(jax.jit, static_argnums=(0,))
    def f_single(self,v):
        I_hires = jax.image.scale_and_translate(self.data, #data=psf, I = image high resolution
                                                shape = (self.N_crop_os, self.N_crop_os, 1), #shape of images to be interpolated
                                                spatial_dims = (0, 1, 2), 
                                                scale = jnp.array([1, 1, 1]), 
                                                translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                                method = self.interp_method)[...,0]
        if self.os > 1:
            I_hires = binning(I_hires, self.os) # downsample the interpolated image

        return v[4] + v[3] * I_hires # bg + signal * image
        

    @partial(jax.jit, static_argnums=(0,))
    def f_biplane(self,v):
        half_PSF = self.data.shape[1] // 2
        
        I_hires_left = jax.image.scale_and_translate(self.data[:, 0 : half_PSF, :],
                                        shape = (self.N_crop_os, self.N_crop_os, 1),
                                        spatial_dims = (0, 1, 2), 
                                        scale = jnp.array([1, 1, 1]), 
                                        translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]),
                                        method = self.interp_method)[...,0]
                                        
        I_hires_right = jax.image.scale_and_translate(self.data[:, half_PSF : , :], 
                                        shape = (self.N_crop_os, self.N_crop_os, 1),
                                        spatial_dims = (0, 1, 2), 
                                        scale = jnp.array([1, 1, 1]), 
                                        translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]),
                                        method = self.interp_method)[...,0]                                        
        if self.os > 1:
            I_hires_left = binning(I_hires_left, self.os)
            I_hires_right = binning(I_hires_right, self.os)

        return v[4] + jnp.concatenate((v[3]/2.*I_hires_left, v[3]/2.*I_hires_right), axis=1)  
    

    @partial(jax.jit, static_argnums=(0,))
    def generate_image(self, a, v=[0, 0, 0, 1, 0], M=None, z_def=None):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], a parameter vector v=[x,y,z,s,bg], an image M and the defocus
        If Image M is not given, the photon number may be not accurate
        If defocus is not given, the one in the class is used
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class

        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0))
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # calculate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return v[4] + sig_scale * v[3] * image




    # FIELD CALCULATION
    #   - calc_parameters
    #   - fresnel_coefs
    #   - BFP_fields
    #   - czt2_coefs
    #   - czt2
    #   - generate_psf
    #   - gaussian_kernel
    #   - gaussian_filter


    def parameters(self):
        '''This function computes the parameters needed in the psf class '''
        
        # Refractive Indices of: sample, intermediate layer, coverslip         
        if len(self.RI) == 1:
            self.RI = [self.RI[0], self.RI[0], self.RI[0]] # medium 1 ( typ water), 2 ( intermediate layer, when present), 3 (immersion medium=glass)
        elif len(self.RI) == 2:
            self.RI = [self.RI[0], self.RI[1], self.RI[1]] 

        # array of z slices: is the emitters z positions. used to calculate the PSF
        self.z_vec = jnp.linspace(self.z_min, self.z_max, int((self.z_max - self.z_min) / self.uz + 1)) # meters

        '''k-space parameters'''
        self.k0 = 2*jnp.pi/self.wavelength                                  # Wavevector in vacuum
        self.k1, self.k2, self.k3 = [self.k0 * RI for RI in self.RI]        # Wavevectors in media 1, 2 and 3
        self.uk = 2*self.k0 * self.NA / self.Nk                             # BFP pixel size
        k_vec = (jnp.fft.fftfreq(self.Nk, 1/self.Nk)+0.5)*self.uk
        self.Kx, self.Ky = jnp.meshgrid(k_vec, k_vec) 
        self.Kr = jnp.sqrt(self.Kx**2 + self.Ky**2)

        '''parameters for the pupil plane'''
        self.N_pad = jnp.round(2*jnp.pi/self.uk/self.ux)  # number of pixels to pad the BFP fields to meet the resolution ux
        # self.pupil = jnp.fft.ifftshift((zernike_noll(1, self.Nk))).astype(int) # pupil defined by zernike 1
        self.pupil = self.Kr <= (self.k0*self.NA)       # UAF + SAF pupil mask (in fourier space)
        self.pupil_UAF = self.Kr <= (self.k0 * self.NA * (self.RI[0] / self.RI[2])) # UAF pupil mask (in fourier space)

        '''parameters for interpolation'''
        self.N_crop = self.Nx - 4                           # size of images to interpolate
        self.N_crop_os = self.N_crop * self.os              # images size in the oversampled PSF
        self.shift = ((self.Nx-self.N_crop)//2)*self.os     # shift in x and y between the PSF and the image center
        


    #@partial(jax.jit, static_argnums=(0,))
    def fresnel_coefs(self):
        ''' 
        This function computes those Fresnel coefficients that are the same for all the emitters. 
            We consider here the system shared properties:
            - RI: refractive indices of the three media: [RI_sample, RI_intermediate, RI_coverslip];
            - d2: thickness of intermediate layer (medium 2) in meters; (if present)
            - T: field transmission; two pages: for s and p-pol: T=[T_s, T_p];
            We do not consider here the individual emitters properties:
            - dipole: [theta, phi] = polar and azimuthal angles of emitting dipole;
            - mu: magnitude of dipole;
            - z: z position of the emitter dipole;
        '''

        T_s, T_p = self.T[:, :, 0], self.T[:, :, 1]

        # Coordinates in the objective pupil: Eq. 17.
        self.PHI3 = jnp.arctan2(self.Kx, self.Ky)
        #self.THETA1 = jnp.arccos(jnp.sqrt(1-(self.RI(2)/self.RI(0)*self.Kr/k3)**2))   #angle in medium 1
        #self.THETA2 = jnp.arccos(jnp.sqrt(1-(self.RI(2)/self.RI(1)*self.Kr/k3)**2))   #angle in medium 2
        self.THETA3 =jnp.arcsin(self.pupil * self.Kr / self.k3)  #angle in medium 3

        'Snell s law: Eq. 4.'
        # Cos(theta) becomes imaginary when the angle is larger than the critical angle: this is the exponential decay of the evanescent wave.
        self.CTHETA1 = jnp.sqrt(0j + 1 - (self.Kr / self.k1)**2) #this is required to estimate the z_factor
        self.CTHETA2 = jnp.sqrt(0j + 1 - (self.Kr / self.k2)**2)
        self.CTHETA3 = jnp.cos(self.THETA3)

        'Fresnel-coefs: Eq. 3.'

        'transission and reflection coefficients from sample to intermediate layer (medium 1 to medium 2)'
        tp12 = 2 * self.RI[0] * self.CTHETA1 / (self.RI[0] * self.CTHETA2 + self.RI[1] * self.CTHETA1)
        ts12 = 2 * self.RI[0] * self.CTHETA1 / (self.RI[0] * self.CTHETA1 + self.RI[1] * self.CTHETA2)
        rp12 = (self.RI[1] * self.CTHETA1 - self.RI[0] * self.CTHETA2) / (self.RI[0] * self.CTHETA2 + self.RI[1] * self.CTHETA1)
        rs12 = (self.RI[0] * self.CTHETA1 - self.RI[1] * self.CTHETA2) / (self.RI[0] * self.CTHETA1 + self.RI[1] * self.CTHETA2)

        'transission and reflection coefficients from intermediate layer to coverslip (medium 2 to medium 3)'
        tp23 = 2 * self.RI[1] * self.CTHETA2 / (self.RI[1] * self.CTHETA3 + self.RI[2] * self.CTHETA2)
        ts23 = 2 * self.RI[1] * self.CTHETA2 / (self.RI[1] * self.CTHETA2 + self.RI[2] * self.CTHETA3)
        rp23 = (self.RI[2] * self.CTHETA2 - self.RI[1] * self.CTHETA3) / (self.RI[1] * self.CTHETA3 + self.RI[2] * self.CTHETA2)
        rs23 = (self.RI[1] * self.CTHETA2 - self.RI[2] * self.CTHETA3) / (self.RI[1] * self.CTHETA2 + self.RI[2] * self.CTHETA3)

        'From them, we calculate the transmission and reflection Fresnel coefs for the three-layer system: Eq. 12.'
        self.tp = (tp12 * tp23 * jnp.exp(1j * self.k2 * self.d2 * self.CTHETA2)) / (1 + rp12 * rp23 * jnp.exp(2j * self.k2 * self.d2 * self.CTHETA2))
        self.ts = (ts12 * ts23 * jnp.exp(1j * self.k2 * self.d2 * self.CTHETA2)) / (1 + rs12 * rs23 * jnp.exp(2j * self.k2 * self.d2 * self.CTHETA2))

        'Prefactor C (without z_factor): (the only constant where focal_len plays a role). Eq. 11.'
        C = (self.k3**2 * jnp.exp(1j * self.k3 * self.focal_len) * self.CTHETA3) / self.focal_len / self.RI[0] * jnp.exp(-1j * self.k3 * self.d2 * self.CTHETA3 * 0) # correction to put z=0 at the coverslip-intermediate layer interface
        self.C = C / self.k3**1  # NOTE: re-normalization here, to prevent large numbers

        'Apodization of the objective: Eq. 16.'
        # rotation of rays by their angle theta3, such that they are all parallel to the optical axis
        self.Apo_p = 1 / jnp.sqrt(self.CTHETA3) * self.pupil * T_p  # Apodization of the objective lens, T_p is measured field transmission
        self.Apo_s = 1 / jnp.sqrt(self.CTHETA3) * self.pupil * T_s  # Apodization of the objective lens, T_s is measured field transmission



    #@partial(jax.jit, static_argnums=(0,))
    def BFP_fields(self, dipole, mu=1):
        '''
        Calculates the BFP fields (Ex,Ey) of an arbitrary oriented dipole at a distance z from the coverslip.
        It includes also near-field effects (SAF emission). Based on the paper of Axelrod, J. of Microscopy 2012.
            Arguments:
            -  dipole:      [theta, phi] = polar and azimuthal angles of emitting dipole.
            -  mu:          magnitude of dipole: molecule intensity.
        '''

        theta_mu, phi_mu = dipole # orientation of dipole emitter mu
    
        'Dipole projections onto directions s, p and z: Eq. 13.'
        mu_p = mu * jnp.sin(theta_mu) * jnp.cos(phi_mu - self.PHI3)
        mu_s = mu * jnp.sin(theta_mu) * jnp.sin(phi_mu - self.PHI3)
        mu_z = mu * jnp.cos(theta_mu)

        'Field magnitudes in layer 3 (pre-objective zone), along the s, p and z-axes (see paper for axes definitions). Eq. 10.'
        E3p = self.C * self.tp * self.CTHETA3 * (mu_p / self.RI[2] + mu_z * jnp.sin(self.THETA3) /  self.RI[0] / self.CTHETA1)
        E3s = self.C * self.ts * (mu_s / self.RI[2] / self.CTHETA1)
        E3z = self.C * self.tp * jnp.sin(self.THETA3) * (mu_p / self.RI[2] + mu_z * jnp.sin(self.THETA3) /  self.RI[0] / self.CTHETA1)

        'Influence of the objective: rotation of rays by their angle theta3, such that they are all parallel to the optical axis'
        E_BFP_p = (E3p * self.CTHETA3 + E3z * jnp.sin(self.THETA3)) * self.Apo_p
        E_BFP_s = E3s * self.Apo_s  # s-polarization remains unchanged by this rotation

        'Coordinate transform into x-and y-polarization -> fields in the back focal plane of objective'
        E_BFP_x = ((jnp.cos(self.PHI3)) * E_BFP_p - (jnp.sin(self.PHI3)) * E_BFP_s)  
        E_BFP_y = ((jnp.sin(self.PHI3)) * E_BFP_p + (jnp.cos(self.PHI3)) * E_BFP_s)

        return E_BFP_x, E_BFP_y




    def czt2_coefs(self):
        """ This function computes all the terms needed for the 2d chirp z-transform that are common to all the emitters. """
        
        ux_des = self.ux / self.os # desired resolution in nm: effective pixel size in the camera
        self.Nx_os = self.Nx * self.os # x-y size of oversampled PSF

        self.N_pad = 2 ** int(jnp.ceil(jnp.log2(self.Nx_os + self.Nk - 1)))
        self.pad_left = int(jnp.ceil((self.N_pad - self.Nk)/2))
        self.pad_right = (self.N_pad - self.Nk) // 2
        self.start_idx = int(jnp.ceil((self.N_pad - self.Nx_os)/2))
        self.end_idx = self.start_idx + self.Nx_os
        x = jnp.arange(-self.N_pad // 2 , self.N_pad // 2)

        ux_fft = 2 * jnp.pi / (self.Nk * self.uk)  # FFT resolution without padding
        r = ux_des / ux_fft  # Required r-factor to meet the desired resolution at the given grid size N: this is the advantage of czt2 transform
        self.alpha = r / self.Nk # alpha = ux * uk / os / 2 / pi
        self.kernel = jnp.exp(-1j * self.alpha * jnp.pi * x[:, jnp.newaxis]**2) * jnp.exp(-1j * self.alpha * jnp.pi * x**2)
        self.fft_kernel = jnp.fft.fft2(jnp.fft.ifftshift(self.kernel))




    def czt2(self, E_in):
        """ 2D Chirp Z-Transform. Used to compute the Fourier transform of the BFP: propagates the fields from the BFP to the camera."""
        'The benefit of the chirp transform is that the size of the pupil function is not dependent on the pixel'
        'size of the camera and a pupil size of 128 x 128 pixels can represent a reasonable sampling in the pupil function.'
         
        E_in = jnp.conj(E_in) #note that czt2 seems to invert the phase compared to fft2! this is a quick fix for now.
        # NOTE: the final conjugation seems to be necessary (compared with Nicolas simulations), but so far, I don't know why!
        
        f1 = jnp.pad(E_in, ((0,0), (self.pad_left, self.pad_right), (self.pad_left, self.pad_right)), mode='constant')
        f1 *= jnp.conj(self.kernel[jnp.newaxis, :,:])
        f1 = jnp.fft.fft2(jnp.fft.ifftshift(f1, axes=(1,2)), axes = (1,2))
        f1 *= self.fft_kernel
        f1 = jnp.fft.fftshift(jnp.fft.ifft2(f1, axes=(1,2)), axes=(1,2)) * self.alpha * jnp.conj(self.kernel)
        E_out = f1[:, self.start_idx:self.end_idx, self.start_idx:self.end_idx] #crop the image to match Nx*os

        return E_out


    def generate_psf(self, emitter_z_pos, exp_jph):
        """ 
        This function calculates the intensity image at the sensor of a pointlike emitter at a given z position.
        z = 0 is at the coverslip-sample interface, with z positive moving in the sample.
        """
        '''z dependent phase factor'''
        # Field propagation in the sample. (only factor where z position is considered)
        # z_factor = exp(i*z*k1*cos(theta1)) = exp(i*z*k1*sqrt(1-(kr/k1)^2) ). k1 is the sample's wavevector.
        z_factor = jnp.exp(1j * self.k1 * self.CTHETA1 * (emitter_z_pos[jnp.newaxis, : , jnp.newaxis, jnp.newaxis] - self.d2)) # correction to change z=0 to the coverslip-intermediate layer interface
        self.E_BFP_z = self.E_BFP * z_factor #make the fields 3 dimensional


        '''PSF'''
        # I = sum(|E|**2) over all fields, for every z slice. I == PSF.
        # the czt2 function takes a BFP field (dim = Nk) and returns an ovesampled PSF slice (dim = Nx*os)
        E_fields = jnp.array([self.czt2(jnp.fft.fftshift(field * exp_jph, axes = (1,2))) for field in self.E_BFP_z]) # fields in the camera.                                
        I = jnp.sum(jnp.abs(E_fields)**2, axis=0).transpose(1,2,0) # total intensity in the camera, not normalized.
        I_BFP = jnp.sum(jnp.abs(self.E_BFP_z * self.apodization)**2, axis=0) #total intensity in the BFP, used below in the normalization.
        I /= jnp.sum(I_BFP, axis = (1,2)) # total intensity in the camera, normalized to intensity in the BFP

        '''Electric_fields'''
        E_fields = E_fields.transpose(0, 2, 3, 1) / jnp.sqrt(jnp.sum(I_BFP, axis=(1,2))[None])

        '''Check intensity normalization'''
        # Uncheck this lines to plot the total normalized intensity in every z slide (normalization: BFP = 100% of photons -> sensor = 70%-95%).
        # print(' - z slice (nm) - \t - I (cam / BFP) -')
        # [print(' ', f'{np.round(value  * 1e9):.0f}', '\t\t', ' ',  jnp.sum(I, axis = (0,1))[i]) for i, value in enumerate(z)]
        
        '''BLURRING'''
        if self.sigma_blurring != 0: 
            vmap_blurring = jax.vmap(lambda z_slice: self.blurr_image(z_slice), in_axes=2, out_axes=2)
            I = vmap_blurring(I)
        
        return I, E_fields
    

    def gaussian_kernel(self):
        """Generate the gaussian kernel to blur the PSF. blurring given by sigma."""
        size = int(2 * jnp.ceil(3 * self.sigma_blurring) + 1)
        x = jnp.arange(-size // 2 + 1, size // 2 + 1)
        kernel = jnp.exp(-0.5 * (x / self.sigma_blurring) ** 2)
        kernel = kernel / jnp.sum(kernel)
        self.gauss_kernel = jnp.outer(kernel, kernel)
        


    def blurr_image(self, image):
        """Blurr the image with a Gaussian filter. Intensity does not changes."""
        padded_image = jnp.pad(image, ((2*self.shift, 2*self.shift), (2*self.shift, 2*self.shift)), mode='edge') # pad to remove border artifcats when blurring
        blurred = jsp.signal.convolve(padded_image, self.gauss_kernel, mode='same', method='fft')[2*self.shift:-2*self.shift, 2*self.shift:-2*self.shift]
        normalization_factor = jnp.sum(image) / jnp.sum(blurred) # normalize to initial image intensity
        return blurred * normalization_factor





    # PUPIL FUNCTIONS
    #   - zernikes
    #   - aberrations
    #   - calc_defocus
    #   - calc_SA
    #   - RI_mismatch_correction
    #   - apodization
    #   - calc_CRLB



    def zernikes(self):
        ''' Creates the basis set of Zernike polynomials for the given Noll indices. '''
        n_modes = len(self.Z_modes)
        modeset = np.ones((n_modes, self.Nk, self.Nk)) # initialize modeset basis stack (np array here)
        for (n,zernike) in enumerate(self.Z_modes):
           modeset[n,:,:] = zernike_noll(zernike, self.Nk) 
        self.modeset = jnp.asarray(modeset)
        #self.modeset_fft = jnp.fft.ifftshift(self.modeset)



    def aberrations(self):
        ''' Create pupil aberrations phase given the Noll basis modeset. '''
        aberr = jnp.sum(self.modeset * self.Z_magn[:, jnp.newaxis, jnp.newaxis], axis = 0)
        return jnp.fft.ifftshift(aberr)



    def calc_defocus(self, z_defocus, RI):
        '''
        Generate spherical pupil defocus for a given nominal focal plane z_defocus and refractive index RI.
        z_defocus is the distance between the coverslip and the nominal objective focal plane (the one that you set with the Z piezo).
        IMPORTANT: the PSF distrotiona due to a sample with a different RI are implicitly included in the BFP E_fields though the fresnel coefficients.
            - RI = immersion medium. Used to generate the PSF.
            - RI = sample. Added to correct the PSF focus if desired, through the functions calc_SA and RI_mismatch_correction.
        '''
        defocus =  + z_defocus * jnp.sqrt(0j + (self.k0 * RI)**2 - self.Kr**2) * self.pupil # plus sign here in front of z_defocus
        relative_defocus = defocus - jnp.mean(defocus[self.pupil]) #subtracting offset, for offset free integration
        return defocus, relative_defocus



    def calc_SA(self, z_defocus): 
        '''
        Calculates the spherical aberrations induced from the refractive for an objective with a given nominal defocus z_defocus.
        SA: corrects the PSF distortion and focus position: effective focus and nominal focus will coincide.
        SA_pure: restores only the PSF distortion: effective focus and nominal focus will NOT coincide.
        '''
        _, self.def_sample = self.calc_defocus(z_defocus, self.RI[0])               # defocus in sample (RI = buffer)
        _, def_imm_medium = self.calc_defocus(z_defocus, self.RI[2])                # defocus in immersion medium (RI = oil)
        self.SA = (self.def_sample - def_imm_medium)   #* self.pupil        # this is the full aberration caused by the refractive index mismatch
        self.SA_pure = (self.SA - (jnp.sum(self.SA[self.pupil] * self.def_sample[self.pupil]) / jnp.sum(self.def_sample[self.pupil]**2)) * self.def_sample) * self.pupil #pure spherical aberration -> the defocus term is removed
    




    def RI_mismatch_correction(self, correct_RI_mismatch):
        '''Sets the desired spherical aberration correction'''
        correction_options = {'yes': self.SA, 'pure': self.SA_pure, 'no': 0}
        if correct_RI_mismatch not in correction_options:
            print('--- WARNING --- ')
            print('The RI fluid/coverslip mismatch generates spherical aberrations that move the objective focal position and distort the PSF shape')
            print('Do you want to correct the PSF? type one:')
            print(' - yes:   introduces counter spherical aberrations that restore the PSF shape and the focus z position')
            print(' - pure:  introduces counter spherical aberrations that restore only the PSF shape')
            print(' - no:    simulates the real PSF with spherical aberrations (default option)')
            print('---------------')
            print()
            self.tot_defocus = self.obj_defocus
        else:
            self.tot_defocus = self.obj_defocus + correction_options[correct_RI_mismatch]




    def apodization(self) :
        """ Consider amplitude apodization transmitted by the objective and from the pupil to the camera"""

        _, _, R, pupil = create_coord(self.Nk, 2 / self.Nk, 'exact')

        # Polarization dependent Field transmission from Sample to BFP. 
        Ts = jnp.polyval(self.Ts_coefs, R) * pupil
        Ts = jnp.array(Ts) / jnp.max(Ts)
        Tp = jnp.polyval(self.Tp_coefs, R) * pupil
        Tp = jnp.array(Tp) / np.max(Tp)
        self.T = jnp.fft.ifftshift(jnp.dstack((Ts, Tp)))

        # Effective power transmission from Pupil to Camera.
        T_fit = jnp.polyval(self.T_fit_coefs, R) * pupil
        T_fit = jnp.array(T_fit) / np.max(T_fit)
        #T_fit[T_fit<0] = 0
        self.apodization = jnp.fft.ifftshift(T_fit)




    def calc_CRLB(self, cam, s = 2000, bg = 100, z_slice = 'all'):
        ''' Computes the CRLB of the whole PSF if z = 'all'. Otherwise only at the given z position.'''

        if isinstance(self.z_defocus, list): #biplane imaging
            #split the PSF model into the two channels, calculate Fisher Info for each channel and sum them
            h1 = copy.deepcopy(self)
            h1.data = self.data[:,0:self.data.shape[1] // 2,:]  # take left psf
            h2 = copy.deepcopy(self)
            h2.data = self.data[:,self.data.shape[1] // 2:,:]  # take right psf
            
            FI = Fisher(h1, cam, s/2, bg/2) + Fisher(h2, cam, s/2, bg/2) #calculating Fisher information matrix

        else: #single channel imaging

            FI = Fisher(self, cam, s, bg, z_slice) #calculating Fisher information matrix

        F_inv = jnp.linalg.inv(FI.transpose(2,0,1))  # calculate the inverse of the Fisher information matrix
        CRLB = jnp.array([jnp.sqrt(F_inv[:, i, i]) for i in range(5)]) # calculate Cramer-Rao lower bound
        
        return CRLB






    # PLOT FUNCTIONS
    #   - show_slice_xy
    #   - show_projection
    #   - show_pupil
    #   - show_CRLB


    def show_slice_xy(self):
        """PLot PSF Z slices with imshow"""

        biplane = isinstance(self.z_defocus, list)
        plt.figure(figsize=(6,8)) if biplane else plt.figure(figsize=(10,3)); 
        fs = 12 

        y_step = (self.Nx-1) / 4
        x_step = (2*self.Nx-1) / 4 if biplane else y_step
        y_ticks = ((2 * np.arange(5) * y_step + 1) * self.os -1) / 2
        x_ticks = ((2 * np.arange(5) * x_step + 1) * self.os -1) / 2
        y_ticklabels = (np.arange(5) * y_step + 1).astype(int) # tick labels
        x_ticklabels = 2 * y_ticklabels  if biplane else y_ticklabels # tick labels   

        z_defocus_str = f'{self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f}' if biplane else f'{self.z_defocus*1e9:.0f}'
        plt.suptitle(f'Objective focused at {z_defocus_str} nm', fontsize=fs)

        for i, z_pos in enumerate([0, int(self.data.shape[2] // 2 + 1), int(self.data.shape[2] - 1)]):
            plt.subplot(3, 1, i + 1) if biplane else plt.subplot(1, 3, i + 1)
            z_label = (self.z_min + (self.z_max - self.z_min) * i / 2) * 1e9
            plt.title(f"{z_label:.0f} nm", fontsize=fs)
            plt.imshow(self.data[..., z_pos] * self.os**2, cmap=parula)
            plt.xticks(x_ticks, x_ticklabels)
            plt.yticks(y_ticks, y_ticklabels)
            plt.xlabel('x / pixels') if not biplane or i == 2 else None
            plt.ylabel('y / pixels') if biplane or i == 0 else None
            plt.colorbar()
        plt.tight_layout()
        plt.show()



    def show_projection(self):
        biplane = isinstance(self.z_defocus, list)
        plt.figure(figsize=(8,6)); fs = 12
        x_limit, y_limit = 400., 140.

        projection_y = np.vstack([np.sum(self.data[:, i * self.data.shape[1]//2:(i+1) * self.data.shape[1]//2]*self.os, axis=1) for i in range(2)]) if biplane else np.sum(self.data*self.os, axis=1)
        projection_x = np.sum(self.data * self.os, axis=0)
        projections = [projection_x, projection_y]
        titles = ['X projection', 'Y projection']
        norm = Normalize(vmin=np.min(projections), vmax=np.max(projections))

        step = (self.Nx-1) / 4
        y_step = y_limit / self.Nx
        y_ticks = y_step * (np.arange(5) * step + 0.5)  # y tick positions
        y_ticklabels = (np.arange(5) * step + 1).astype(int)  # tick labels
        y_ticklabels = 2 * y_ticklabels if biplane else y_ticklabels  # tick labels

        x_ticks = np.arange(5) * x_limit / 4
        x_ticklabels = [int(((self.z_max - self.z_min) * i / 4 + self.z_min) * 1e9) for i in range(5)]

        # plot projections
        for i, (title, projection) in enumerate(zip(titles, projections), 1):
            plt.subplot(2, 1, i)
            plt.imshow(projection, norm=norm, extent=[0, x_limit, y_limit, 0], cmap=parula)
            plt.title(title, fontsize=fs)
            plt.ylabel('x / pixels') if i == 1 else plt.ylabel('y / pixels')
            plt.xticks(x_ticks, x_ticklabels)
            plt.yticks(y_ticks, y_ticklabels)
            # plt.colorbar()

        z_defocus_str = f'{self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f}' if biplane else f'{self.z_defocus*1e9:.0f}'
        plt.suptitle(f'Objective focused at {z_defocus_str} nm', fontsize=fs)
        plt.xlabel('z / nm')
        plt.tight_layout()
        plt.show()




    def show_pupil(self, pupil='SAF'):

        pupil = self.pupil_UAF if pupil == 'UAF' else self.pupil # for pupil far from the coverslip: UAF, else: SAF
        plt.figure(figsize=(10, 3)); fs = 12

        biplane = isinstance(self.z_defocus, list)
        titles = ["Aberrations L channel", "Aberrations R channel", "Apodization"] if biplane else ["Aberrations", "Defocus", "Apodization"]
        cmaps = ['hsv', 'hsv', 'gray'] if biplane else ['hsv', 'plasma', 'gray']

        first_plot = np.real(jnp.fft.fftshift(self.pupil_aberrations[0] * pupil)) if biplane else np.real(jnp.fft.fftshift(self.pupil_aberrations * pupil))
        second_plot = np.real(jnp.fft.fftshift(self.pupil_aberrations[1] * pupil)) if biplane else np.real(jnp.fft.fftshift(jnp.real(-self.tot_defocus) * pupil))
        third_plot = jnp.fft.fftshift(self.apodization * pupil)
        pupils = [first_plot, second_plot, third_plot]

        for i, (title, cmap, pupil) in enumerate(zip(titles, cmaps, pupils), 1):
            plt.subplot(1, 3, i)
            plt.title(title, fontsize=fs)
            plt.imshow(pupil, cmap=cmap)
            plt.xticks([]); plt.yticks([])
            plt.colorbar()
        plt.suptitle(f'Pupil  ', fontsize=fs+3)
        plt.tight_layout()
        plt.show()




    def show_SA(self):
        plt.figure(figsize=(10, 3)); fs = 12

        titles = ["Real defocus", "RIM corrected", "RIM corrected (pure)"]
        first_plot = np.real(jnp.fft.fftshift(jnp.real(-self.obj_defocus) * self.pupil))
        second_plot = np.real(jnp.fft.fftshift(jnp.real(-(self.obj_defocus + self.SA) * self.pupil_UAF)))
        third_plot = np.real(jnp.fft.fftshift(jnp.real(-(self.obj_defocus + self.SA_pure)) * self.pupil_UAF))

        pupils = [first_plot, second_plot, third_plot]
        vmin, vmax = np.min([np.min(pupil) for pupil in pupils]), np.max([np.max(pupil) for pupil in pupils])

        for i, (title, pupil) in enumerate(zip(titles, pupils), 1):
            plt.subplot(1, 3, i)
            plt.title(title, fontsize=fs)
            # plt.imshow(pupil, cmap='plasma')
            plt.imshow(pupil, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.colorbar()
        plt.suptitle(f'Defocus', fontsize=fs+3)
        plt.tight_layout()
        plt.show()




    def show_CRLB(self, cam, s=2000, bg=100, ylim = False, export=False):
        x_axis = self.z_vec * 1e9  # z range
        CRLB = self.calc_CRLB(cam, s, bg) * 1e9

        if not isinstance(s, (int, float)): # get  signal and background just for plotting. CRLB have been computed with effective s and bg.
            s /= cam.QE if cam.EM_gain == 1  else  cam.QE / 2
            bg /= cam.QE if cam.EM_gain == 1 else cam.QE / 2

        fig, ax1 = plt.subplots()

        # plot CRLB
        ax1.plot(x_axis, CRLB[0], linestyle='-', label=r'σ$_{x}$')
        ax1.plot(x_axis, CRLB[1], linestyle='-', label=r'σ$_{y}$')
        ax1.plot(x_axis, CRLB[2], label=r'σ$_{z}$')
        ax1.set_xlim(min(x_axis), max(x_axis))
        ax1.set_ylim(0, ylim) if ylim else None
        ax1.set_xlabel('z / nm')
        ax1.set_ylabel('σ / nm')
        ax1.grid()

        defocus_str = f'Objectives focal planes: {self.z_defocus[0]*1e9:.0f} and {self.z_defocus[1]*1e9:.0f} nm' if isinstance(self.z_defocus, list) else f'Objective focal plane: {self.z_defocus*1e9:.0f} nm'
        fig.suptitle(f'Theoretical CRLB')
        fig.text(0.28, 0.85, defocus_str, fontsize=12, color='black', rotation=0)
        fig.text(0.28, 0.8, f'Mean Signal: {np.mean(s):.0f} / molecule', fontsize=12)
        fig.text(0.28, 0.75, f'Mean Background: {np.mean(bg):.0f} / pixel', fontsize=12, color='black', rotation=0)

        # Check if s is a scalar
        if not isinstance(s, (int, float)):
            # Adding the second y-axis if s is not a scalar
            ax2 = ax1.twinx()
            ax2.plot(x_axis, s, color='red', label='s', linestyle='--')
            ax2.plot(x_axis, bg, color='black', label='bg', linestyle='--')
            ax2.set_ylabel('signal / ph', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            # Legend for only the primary y-axis
            ax1.legend(loc='best')

        plt.tight_layout()
        plt.show()

        if export:
            return CRLB[0], CRLB[1], CRLB[2]



# ---------------------- PSF CHILDREN CLASSES ----------------------


# PSF class for fft interpolation

class psf_fft(psf):
    '''Children psf class for FFT interpolation'''

    def __init__ (self, psf_instance):

        self.__dict__.update(psf_instance.__dict__)     # Initialize attributes from the psf instance
        self.shift_fft = (self.Nx - self.N_crop) // 2   # shift in x and y between the PSF and the image center in Fourier space (before os)

    def generate_image_fft(self, param):

        x, y, z, s, bg = param
        exp_xy = jnp.exp(1j * (self.Kx * y * self.ux + self.Ky * x * self.ux))
        exp_z = jnp.exp(1j * self.k1 * self.CTHETA1 * z * self.uz)
        Ez = self.E_BFP * exp_z # pupil * exp_z
        I_tot = jnp.sum(jnp.abs(Ez * self.apodization)**2, axis=0)
        E_fields = jnp.array([self.czt2(jnp.fft.fftshift(field * self.exp_jphase * exp_xy, axes = (-1,-2))) for field in Ez])#.sum(axis=0)
        I_field = jnp.sum(jnp.abs(E_fields)**2, axis=0)
        _nrm = jnp.where(jnp.sum(I_tot) == 0, 0, 1./jnp.sum(I_tot)) # normalization factors
        itp = (I_field * _nrm )[0] # normalized image
        itp = self.blurr_image(itp) if self.sigma_blurring != 0 else itp # blurr image

        I_binned = binning(itp, self.os)
        Image = bg + s * I_binned
        
        return Image
    
    def f(self, v):

        return self.generate_image_fft(v)[self.shift_fft:-self.shift_fft, self.shift_fft:-self.shift_fft]







# PSF class for deconvolution

class psf_deconvolution(psf):
    '''Children psf class for gradient descent deconvolution'''
        
    def __init__(self,
                 psf_instance,
                 method = 'multiplane',
                 z_defocus = 0,
                 planes = 0,
                 photon_statistics = 'poissonian',
                 IF_width = 1e4):
        
        self.__dict__.update(psf_instance.__dict__)         # Initialize attributes from the psf instance
        self.method = method                                # method for the deconvolution: 'multiplane
        self.photon_statistics = photon_statistics          # photons statistics for the error function: 'poissonian' or 'gaussian'
        self.z_defocus = z_defocus * 1e-9                   # defocus in m
        self.planes = (z_defocus + np.array(planes)) * 1e-9 # distance between planes for multiplane deconvolution. in m
        self.n_planes = self.planes.shape[0]                # number of planes
        self.IF_width = IF_width * 1e-9                     # width of the DPP influence function. in m

        '''Initials'''
        self.actuator_IF()                                  # inizialize influence function of the DPP actuator
        self.DPP(actuator_v=np.zeros(self.actuator_no))     # initialize DPP field in the pupil
        self._grad = self.grad_fun()                        # initialize the forward and grad function for the deconvolution
        self.update = get_update(self._grad)                # update function. to call inside the gradient descent while loop


        '''DECONVOLUTION METHODS'''
        '''multiplane: fits aberrations and positions. aberrations are fitted zernike-wise.'''
        '''multiplane_bead: fits aberrations and positions for beads data like multiplnae, but pos is constrained between planes.'''
        '''multiplane_pxl: fits aberrations and positions. aberrations are fitted pupil pixel-wise.'''



    # FUNCTIONS
    #  - compute_err
    #  - grad_fun
    #  - fwd_multiplane
    #  - fwd_multiplane_bead
    #  - fwd_multiplane_pxl
    #  - generate_image_bead
    #  - generate_image_dpp
    #  - actuator_IF
    #  - DPP



    ### Optimization Processes---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def compute_err(self, model, data):
        '''Returns the error function used in the deconvolution'''

        if self.photon_statistics == 'poissonian':  # for low photon numbers (SMLM data)
            return compute_LLR(model, data)
        elif self.photon_statistics == 'gaussian':    # for high photon numbers (Beads data)
            return compute_least_squares(model, data)
        else:
            raise ValueError(f"Unsupported photon statistics: {self.photon_statistics}. Choose 'poissonian' for LLR or 'gaussian' for Least Squares.")


    def grad_fun(self):
        '''Returns the forward function used in the deconvolution'''
        if self.method == 'multiplane':
            return jax.value_and_grad(self.fwd_multiplane)
        if self.method == 'multiplane_bead':
            return jax.value_and_grad(self.fwd_multiplane_bead)
        elif self.method == 'multiplane_pxl':
            return jax.value_and_grad(self.fwd_multiplane_pxl)
        else:
            raise ValueError(f"Unsupported method: {self.method}. Choose 'multiplane', 'dpp_pxl'.")
    

    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane(self, par_l, const_l):
        '''Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase.'''
        a, v_v_l = par_l      # set of variables that are updated
        M_v_l = const_l       # set of constants that are not updated
        error = 0             # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda v, M: self.generate_image(a, v, M, self.planes[i]))(v_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error


    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_bead(self, par_l, const_l):
        '''
        Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase
         - pos is the bead position array. different among different beads but constant among the same bead at different planes.
         - ph_v_l is the signal and bg list of arrays. independent among different beads and planes.
        .'''
        a, pos, ph_v_l = par_l  # set of variables that are updated
        M_v_l = const_l         # set of constants that are not updated
        error = 0               # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda ph, M: self.generate_image_bead(a, pos, ph, M, self.planes[i]))(ph_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error


    @partial(jax.jit, static_argnums=(0,))
    def fwd_multiplane_pxl(self, par_l, const_l):
        '''Computes the cumulative error for all the images in all the planes, using the same pupil aberrations and dpp phase.'''
        dpp_phase = par_l           # set of variables that are updated
        M_v_l, v_v_l = const_l      # set of constants that are not updated
        error = 0                   # initialize error
        for i in range(self.n_planes): # iterate over the planes
            error += self.compute_err(jax.vmap(lambda v, M: self.generate_image_dpp(dpp_phase, v, M, self.planes[i]))(v_v_l[i], M_v_l[i]), M_v_l[i]).sum()
        return error


    @partial(jax.jit, static_argnums=(0,))
    def generate_image_bead(self, a, pos=[0, 0, 0], ph=[1, 0], M=None, z_def=None):
        """
        Generates a molecule image given a set of zernikes a=[a1,a2,...], and two parameter vectors: pos=[x,y,z] and  ph=[s,bg], an image M and the defocus
        If Image M is not given, the photon number may be not accurate
        If defocus is not given, the one in the class is used
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class

        aberr = jnp.fft.ifftshift(jnp.sum(self.modeset * a[:, jnp.newaxis, jnp.newaxis], axis = 0))
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1]) # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + aberr)) * self.apodization; # phase
        PSF,_ = self.generate_psf(self.z_vec, exp_jph) # calculate 3D psf
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1), #shape of image to be interpolated
                                            spatial_dims = (0, 1, 2), 
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([pos[0]-self.shift, pos[1]-self.shift, -pos[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return ph[1] + sig_scale * ph[0] * image

    @partial(jax.jit, static_argnums=(0))
    def generate_image_dpp(self, dpp_phase=None, v=[0, 0, 0, 1, 0], M=None, z_def=None):
        """
        generates a molecule image given a set of zernikes, parameter vector v=[x,y,z,s,bg], an image M and the defocus
            Arguments:
            - dpp_phase:    fftshifted phase landscape of the dpp (shape=(Nk, Nk)), if None the phase is set to zero for the whole pupil
            - v:            molecule position: [x, y, z, signal, background] (in psf units)
            - M:            images. Used to scale the interpolated photon numbers. If None the fit works but the photons numbers are wrong.
            - z_def:        defocus (m)            
        """

        sig_scale = 1 if M is None else (M - M.min()).sum()
        z_def = self.z_defocus if z_def is None else z_def # if z_def is not given, use the one in the class
        
        dpp = jnp.zeros(self.DPP_phase.shape) if dpp_phase is None else dpp_phase
        dpp = dpp * self.pupil
        defocus,_ = self.calc_defocus( - z_def, self.RI[-1])    # defocus in immersive medium
        exp_jph = jnp.exp(1j*jnp.real(defocus + self.pupil_aberrations + dpp)) * self.apodization               
        PSF,_ = self.generate_psf(self.z_vec, exp_jph)
        image = jax.image.scale_and_translate(PSF, 
                                            shape = (self.N_crop_os, self.N_crop_os, 1),
                                            spatial_dims = (0, 1, 2),
                                            scale = jnp.array([1, 1, 1]), 
                                            translation = jnp.array([v[0]-self.shift, v[1]-self.shift, -v[2]]), # y, x, z
                                            method = self.interp_method)[...,0]
        if self.os > 1:
            image = binning(image, self.os)

        return v[4] + sig_scale * v[3] * image




    def actuator_IF(self):
        """
        Computes the influence function of one actuator
        """

        '''initialize influence function as Gaussian'''
        self.IF_sigma = 8*self.uk #width of influrence function IF
        self.IF_max = self.IF_width * self.uk # max. phase shift of DPP actuator in pupil units
        self.IF = self.IF_max * jnp.fft.fftshift(jnp.exp(-(self.Kr**2)/2/self.IF_sigma**2)) #influence function of a DPP actuator (here Gaussian)
        
        '''Pixels where Influence function acts (positions of the actuators of the DPP)'''
        xpos, ypos = jnp.meshgrid(jnp.arange(-4, 4), jnp.arange(-4, 5)) # hegaonal grid of actuators
        self.scale_hex = self.Nk/7/jnp.sqrt(3)
        self.IF_Y = ypos * 1.5 * self.scale_hex
        self.IF_X =  xpos* jnp.sqrt(3) *self.scale_hex
        self.IF_X = self.IF_X.at[1::2].set(self.IF_X[1::2] + jnp.sqrt(3)/2 * self.scale_hex)

        '''actuators are arranged within a circular iris'''
        self.circular = ((self.IF_X*self.uk)**2+(self.IF_Y*self.uk)**2 <= (self.k0*self.NA)**2) # circular mask # changed here to <= instead of just < to include the border pixels
        self.actuator_no = jnp.count_nonzero((self.circular).astype(float))
        self.circle = (self.circular).astype(float).at[self.circular.astype(float)==0].set(-100000) # exclude pixels outside the DPP by setting them to -100000... ?



    def DPP(self, actuator_v):
        """returns the field of the DPP in the pupil, depending on the control voltage vector v""" 

        DPP_phase = jnp.zeros((2*self.Nk, 2*self.Nk)) # initialize DPP phase
        flatten_circular = self.circular.astype(float).ravel()
        indices = jnp.where(flatten_circular==1, size=len(actuator_v))
        flatten_circular = flatten_circular.at[indices].set(actuator_v)
        for i in range(len(flatten_circular)):
            DPP_phase += (
                flatten_circular[i] *
                jnp.roll(
                    jnp.pad(self.IF, (self.Nk//2, int(jnp.ceil(self.Nk/2)))),
                    (-self.IF_Y.ravel()[i], self.IF_X.ravel()[i]),
                    axis = (0,1)))
                    
        DPP_phase = DPP_phase[self.Nk - self.Nk//2:self.Nk + int(jnp.ceil(self.Nk/2)), self.Nk - self.Nk//2:self.Nk + int(jnp.ceil(self.Nk/2))]
        pupil_dpp = jnp.fft.ifftshift(jnp.exp(1j * DPP_phase)) * (self.pupil)
        self.DPP_phase = jnp.fft.ifftshift(DPP_phase) * (self.pupil)
        return pupil_dpp



def get_update(_grad_fun):
    @jax.jit
    def update(par, par_old, lrs, bounds, t1, const):
        '''Nesterof Gradient descent update with momentum and bounds'''
        err, par_bar = _grad_fun(par, const) # par: parameters to update, M: images
        x = jax.tree_map(lambda p, lrs, g, bmin, bmax: jnp.clip(p - lrs * g, bmin, bmax), par, lrs, par_bar, bounds[0], bounds[1]) # par_update
        t2 = 0.5 * (1 + jnp.sqrt(1 + 4 * t1**2)) # momentum step update (Nesterof)
        par_ud = jax.tree_map(lambda x, xo, g: x + (t1-1)/t2 * (x - xo), x, par_old, par_bar) # momentum update
        return err, par_ud, x, t2
    return update









# FUNCTIONS NOT IN CLASS

#   PSF SUPPORT FUNCTIONS
#   - Fisher
#   - add_CRLB
#   - create_coord
#   - load_PSF
#   - combine_PSFs
#   - binning

#   PRELOC FUNCTIONS
#   - prepare_images_for_fitting
#   - draw_peaks
#   - crop_images
#   - filter_biplane_double_peaks
#   - preloc_setup
#   - preloc_and_crop


#   FIT FUNCTIONS
#   - estimate_v0
#   - perform_fit
#   - lm_poisson
#   - compute_LLR
#   - fit_step
#   - add_preloc_positions


#   PLOT FUNCTIONS
#   - show_prelocs
#   - show_results_3d
#   - show_results_2d
#   - show_results_1d
#   - show_results_hist
#   - print_results
#   - filter_results



#@partial(jax.jit, static_argnames=('h'))
def Fisher(h, cam, s, bg, z_slice = 'all'):
    ''' 
    Computes the Fisher information matrix. If z_slice = 'all' : whole PSF. Otherwise only at the given z position.
    model:              I = bg + s * PSF(x,y,z)     
    gradient:           dI/dx = s * d(PSF)/dx
                        dI/ds = PSF
                        dI/dbg = 1
    matrix elements:    FI_ij = integral(dI/di * dI/dj)      i,j = variables: x, y, z, s, bg
    '''
    
    s *= cam.QE; bg *= cam.QE  # we consider photons in the BFP, thus we multiply by the QE

    # EMCCDs suffer from excess noise: superpoissonian statistics. this is taked into account by dividing the signal and bg by a factor of 2
    if cam.EM_gain != 1:
        s /= 2; bg /= 2

    if z_slice == 'all':
        # Take all z_slices of the PSF
        Nz = h.data.shape[2]
        data = jnp.stack([binning(h.data[:,:,m], h.os) for m in range(Nz)], axis=-1)
        Dy_px, Dx_px, Dz_px = jnp.gradient(data)   # y = row, x = column

    else: 
        # If z is a number, take only one z slice of the PSF
        Nz = 1
        data_temp = jnp.stack([binning(h.data[:,:,m], h.os) for m in jnp.arange(z_slice - 1, z_slice + 2, 1)], axis=-1)
        
        data = data_temp[..., 1] # use 1 slice to compute the gradient in x and y
        Dy_px, Dx_px = jnp.gradient(data)   # y = row, x = column
        Dz_px = jnp.gradient(data_temp, axis=2)[..., 1] # use 3 slices to compute the gradient in z

    Dx = s * Dx_px / h.ux       # Dx = dI/dx = s * d(PSF)/dx
    Dy = s * Dy_px / h.ux       # Dy = dI/dy = s * d(PSF)/dy
    Dz = s * Dz_px / h.uz       # Dz = dI/dz = s * d(PSF)/dz
    Ds = data                   # Ds = dI/ds = PSF
    Db = jnp.ones_like(Dx)      # Db = dI/db = 1
    Di = [Dx, Dy, Dz, Ds, Db]

    # noise = variance. 
    # Variance = (standard deviation) ** 2 of signal + background + readnoise
    noise = jnp.array(s * Ds + bg + cam.readnoise**2) # for poissonians, standard deviation is sqrt(s), sqrt(bg) and readnoise
    
    FI = jnp.zeros((5, 5, Nz))  # initializing Fisher information matrix 
    for i in range(5):
        for j in range(i, 5):
            if i == j: 
                FI = FI.at[i, j, :].set(jnp.sum(Di[i]**2 / noise, axis=(0, 1)))  # Diagonal elements
            else: 
                FI_ij = jnp.sum(Di[i] * Di[j] / noise, axis=(0, 1))  # Off-diagonal elements
                FI = FI.at[j, i, :].set(FI_ij)
                FI = FI.at[i, j, :].set(FI_ij)
    return FI





# get the psf z slice of a given z value in nm
def vmap_get_z_slice(h, z_values):
    z_vec = jnp.round(h.z_vec * 1e9)  # psf z axis
    z_values = jnp.array(z_values)  # input z values in nm

    def get_z_slice(z_start):
        abs_diff = jnp.abs(z_vec - z_start)
        z_idx = jnp.argmin(abs_diff)  # index of the closest z in psf (nm)
        return z_idx
    
    return jax.vmap(get_z_slice)(z_values)





def add_CRLB(h, cam, results, show_progress=True):
    ''' Compute CRLBs of single localizations and append them to the results table as new columns '''

    print('--- Computing individual CRLBs ---')
    print('Localizations:', len(results))
    print('CRLBs computed:')
    CRLB_x,CRLB_y,CRLB_z, = [],[],[]
    size, counter = len(results) // 10, 0
    [z_min, z_max] = h.z_min*1e9, h.z_max*1e9
    z_slices = vmap_get_z_slice(h, results['z'].values)

    for i in range(len(results)) :
        if (i + 1) % size == 0:
            counter += 10
            print(' ', counter, '%')
 
        # skip if localization did not converged inside the PSF
        if not z_min <= results['z'].iloc[i] <= z_max:
            CRLB_x.append(np.nan)
            CRLB_y.append(np.nan)
            CRLB_z.append(np.nan)
            continue

        CRLB = h.calc_CRLB(cam, 
                           s = results['signal'].iloc[i],
                           bg = results['bg'].iloc[i],
                           z_slice = z_slices[i])
        CRLB_x.append(np.round(CRLB[0][0]*1e9,1))
        CRLB_y.append(np.round(CRLB[1][0]*1e9,1))
        CRLB_z.append(np.round(CRLB[2][0]*1e9,1))

    print('--- Completed ---')
    results_CRLB = results.copy()
    results_CRLB['CRLB x'] = CRLB_x
    results_CRLB['CRLB y'] = CRLB_y
    results_CRLB['CRLB z'] = CRLB_z

    return results_CRLB


def find_z_idx(h, z_start):
    "Returns the index of z_vec closer to z_start (units in nm)"

    z_vec = jnp.round(h.z_vec * 1e9) # psf z axis
    closest_z = int(min(jnp.round(h.z_vec * 1e9), key=lambda x: abs(z_start - x))) # closest z in psf (nm)
    z_idx = jnp.where(z_vec == closest_z)[0][0] # PSF index that corresponds to input z value

    return z_idx


def create_coord(N, u, mode):  
    '''
    creates axis with N entries (if N=[Nx Ny], a 2D grid is created)
    u...spacing between two neighbouring entries
    mode...'exact' or 'FFT'
    if 'exact' is chosen: zero point is exactly in the middle, i.e. for even grid
    sizes between the two middle pixels
    if 'FFT' is chosen: zero point is the middle pixel for odd grid sizes and
    the pixel to the lower right of the exact centre for even grid sizes.
    '''
    if isinstance(N, int):
        N = (N, N)

    if isinstance(u, (float, int)):
        u = (u, u)
 
    if mode == 'exact':
        x = np.linspace(-(N[0] - 1) / 2, (N[0] - 1) / 2, N[0]) * u[0]
        y = np.linspace(-(N[1] - 1) / 2, (N[1] - 1) / 2, N[1]) * u[1]
    elif mode == 'FFT':
        x = (np.arange(-N[0] // 2 + 1, N[0] // 2 + 1)) * u[0]
        y = (np.arange(-N[1] // 2 + 1, N[1] // 2 + 1)) * u[1]

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R**2 <= ((min(N) / 2 * u[0]))**2

    return X, Y, R, mask



def load_PSF(filename):
    """ Load PSF.mat files and construct the classes objective, camera, and PSF """
    """ The PSF stack is imported from matlab, not generated in python """

    PSF = scipy.io.loadmat(filename)['PSF'][0][0]

    wavelength, z_defocus, RI_sample, Nx, ux, uz, os = (
        PSF[i][0][0] * 1e9 if i in [1, 2, 5, 6] else PSF[i][0][0] for i in range(1, 8))

    data, cam, obj = PSF[8], PSF[9][0,0], PSF[10][0,0]

    pixsize, QE, baseline, amp, EM_gain, readnoise = (
        cam[i][0][0] * (1e9 if i == 1 else 1) for i in range(1, 7))

    M, NA, RI = (obj[i][0][0] for i in range(1, 4))
    Ts_coefs, Tp_coefs = (obj[i][0] for i in range(4, 6))

    Z_modes, Z_magn, T_fit_coefs = (
        PSF[11][0,0][i][0] for i in range(3))
    
    cam = Camera(pixsize=pixsize, QE=QE, baseline=baseline, EM_gain=EM_gain, amp=amp, readnoise=readnoise)
    obj = Objective(M=M, NA=NA, RI=RI, Ts_coefs=Ts_coefs, Tp_coefs=Tp_coefs)

    Nk = 128
    z_max = (data.shape[2] - 1) * uz
    h = psf(NA=NA, wavelength=wavelength, z_defocus=z_defocus, RI=[RI_sample, RI], Nk=Nk, Nx=Nx, ux=ux, uz=uz, z_max=z_max, Z_modes=Z_modes, Z_magn=Z_magn, Ts_coefs=Ts_coefs, Tp_coefs=Tp_coefs, T_fit_coefs = T_fit_coefs, os=os)
    h.data = data

    return cam, obj, h


def load_aberr(filename, mat_version = None):
    """Load aberr.mat files"""
    if mat_version == '7.3' : 
        aberr = mat73.loadmat(filename)['aberr']
        Z_modes, Z_magn, T_fit_coefs = aberr['Z_modes'], aberr['Z_magn'], aberr['T_coefs']
    else:
        aberr = scipy.io.loadmat(filename)['aberr'][0][0]
        Z_modes, Z_magn, T_fit_coefs = (aberr[i][0] for i in range(3))

    return Z_modes, Z_magn, T_fit_coefs


def combine_PSFs(h1,h2):
    ''' Concatenates 2 psf in one along the x axis for biplane imaging'''
    h = copy.copy(h1)
    h.data = jnp.concatenate((h1.data, h2.data), axis = 1)
    h.z_defocus = [h1.z_defocus, h2.z_defocus]
    h.aberr = [h1.aberr, h2.aberr]
    return h



def binning(image, bin_size):
    """ Bins an image by summing pixel values in blocks of size bin_size x bin_size"""

    height, width = image.shape[:2]
    binned_height, binned_width = height // bin_size, width // bin_size

    # Reshape the image into a 2D array of blocks and sum pixel values in each block
    superpixels = jnp.sum(image[:binned_height * bin_size, :binned_width * bin_size].reshape(
        binned_height, bin_size, binned_width, bin_size), axis=(1, 3))

    return superpixels








  
    

#   PRELOC FUNCTIONS
#   - prepare_images_for_fitting
#   - find_peaks
#   - draw_peaks
#   - crop_images
#   - filter_biplane_double_peaks
#   - preloc_setup
#   - preloc_and_crop




def cts2phs(I_counts, camera):#, file_format='tif'):
    ''' Convert counts to photons'''
    # baseline = 100 if file_format == 'dcimg' else camera.baseline #dcimg default baseline is 100
    I_photons = np.asarray(I_counts, dtype=np.int16)
    I_photons = np.asarray((I_photons - camera.baseline) * camera.amp / camera.QE, dtype=np.float32)  # subtract baseline, convert counts to electrons, convert electrons to photons
    I_photons = np.maximum(I_photons, 0.01)                 # set minimum pixel values to very small photons
    return I_photons




def find_peaks(img, sigma, min_dist):
    """ Find the peaks in the image using the convolution with a gaussian filter and the local maxima algorithm."""
    conv_img = gaussian_filter(img, sigma=sigma)
    peaks = peak_local_max(conv_img, min_distance=min_dist).astype(int)  # xy pixel coordinates (x=columns, y=rows)
    return conv_img, peaks



#@partial(jax.jit, static_argnames=('N_crop'))
def draw_peaks(peaks, N_crop, offset=0, threshold_rel=1, sn_ratio=None):
    ''' Draw peaks on the preloc image with color according to sn_ratio'''

    if sn_ratio is None:
        sn_ratio = np.ones(len(peaks))

    for i, peak in enumerate(peaks) :
        color = 'yellow' if sn_ratio[i] >= threshold_rel else 'red'
        rectangle = patches.Rectangle(
            (peak[1] + offset - N_crop / 2, peak[0] - N_crop / 2), # peak[1] = y coordinate (rows, with origin on the top of the image, like in the camera), peak[0] = x coordinate (column)
            N_crop, N_crop, color=color, fill=False)
        plt.gca().add_patch(rectangle)
        plt.scatter(peak[1] + offset, peak[0], color='red', s=1) # y coordinate (row), x coordinate (column)




#@partial(jax.jit, static_argnames=('N_crop', 'thresh'))
def crop_images(img, tmp_peaks, N_crop, threshold_abs=0, threshold_rel=0):
    ''' Crop the molecules found with find_peaks'''
    
    peaks, cropped_images, sn_ratio, tmp_sn_ratio = [], [], [], []

    for peak in tmp_peaks:
        row_starts, col_starts = np.subtract(peak, N_crop // 2)
        row_ends, col_ends = row_starts + N_crop, col_starts + N_crop

        tmp_image_crop = img[row_starts:row_ends, col_starts:col_ends] # row / column
        max_val, min_val = np.max(tmp_image_crop), np.min(tmp_image_crop)      
        tmp_sn_ratio.append(np.round((max_val / min_val), 2))

        if max_val > min_val * threshold_rel and min_val >= threshold_abs:
            peaks.append(peak)
            cropped_images.append(tmp_image_crop)
            sn_ratio.append(np.round((max_val / min_val), 2))

    return peaks, cropped_images, sn_ratio, tmp_sn_ratio




def merge_double_peaks(peaks_left, peaks_right, sn_ratio_left, sn_ratio_right):
    """ Merges the peaks that are present in both channels. It keeps the position of the one with the higher signal to noise"""
    
    if not peaks_left or not peaks_right: # if there are no peaks in one of the stacks or both are empty
        return peaks_left if peaks_left else peaks_right

    max_dist = 3  # Set the maximum distance threshold
    distances = cdist(peaks_left, peaks_right)  # point by point distance between the 2 stacks
    indices = np.where(distances < max_dist)  # find the indices of the peaks that are closer than max_dist
    idx_left = [indices[0][idx] for idx in range(len(indices[0])) if sn_ratio_left[indices[0][idx]] <= sn_ratio_right[indices[1][idx]]]
    idx_right = [indices[1][idx] for idx in range(len(indices[0])) if sn_ratio_left[indices[0][idx]] > sn_ratio_right[indices[1][idx]]]
    return np.concatenate((np.delete(peaks_left, idx_left, axis=0), np.delete(peaks_right, idx_right, axis=0)), axis=0)








def preloc_setup(img, sigma, min_dist, threshold_abs, threshold_rel, N_crop, figsize=(8,6), vmin=None, vmax=None, biplane=False):
    ''' Find good parameters for the preloc'''
    
    if N_crop % 2 == 0 : # use odd N_crop to have a symmetric cropping around the central pixel
        print('Use N_crop = odd. ')
        return

    # initials
    min_dist = max(min_dist, N_crop // 2 + 2)  # Minimum distance between peaks and image border
    long_axis = img.shape[1] # x axis = number of columns
    short_axis = long_axis // 2 if biplane else long_axis
       

    # plt.figure(figsize=(figsize, figsize - (2* long_axis//short_axis)))
    plt.figure(figsize=figsize)
    print('Pixel range values: [', np.min(img), ',', np.max(img), ']')
    print('S/N ratio (in descending order):', )

    # peaks left channel (if not biplane left channel = only channel)
    img_left = img[:, :short_axis]
    conv_img_left, tmp_peaks_left = find_peaks(img_left, sigma, min_dist)
    peaks_left, _ , sn_ratio_left, tmp_sn_ratio_left = crop_images(img_left, tmp_peaks_left, N_crop, threshold_abs, threshold_rel)
    
    if biplane :
        # peaks right channel
        img_right = img[:, short_axis:]
        conv_img_right, tmp_peaks_right = find_peaks(img_right, sigma, min_dist)
        peaks_right, _ , sn_ratio_right, tmp_sn_ratio_right = crop_images(img_right, tmp_peaks_right, N_crop, threshold_abs, threshold_rel)
        conv_img = np.concatenate((conv_img_left, conv_img_right), axis=1)
        peaks = merge_double_peaks(peaks_left, peaks_right, sn_ratio_left, sn_ratio_right) # delete the same peaks present in both channels
        
        draw_peaks(tmp_peaks_right, N_crop, offset=short_axis, threshold_rel=threshold_rel, sn_ratio = tmp_sn_ratio_right) # plot right stack peaks
        plt.axvline(x=short_axis-1, color='black', linestyle='-', linewidth=0.5) # draw a line between the 2 stacks
        print('L channel:', sorted(tmp_sn_ratio_left, reverse=True))
        print('R channel:', sorted(tmp_sn_ratio_right, reverse=True))

        # plot the convoluted image
        plt.imshow(conv_img, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Convoluted Image'); plt.tight_layout(); plt.colorbar()
        draw_peaks(tmp_peaks_left, N_crop, offset=0, threshold_rel=threshold_rel, sn_ratio=tmp_sn_ratio_left); plt.show()
        
        # plot the image with the good molecules
        plt.figure(figsize=(figsize, figsize - (2* long_axis//short_axis)))
        draw_peaks(peaks, N_crop)
        draw_peaks(peaks, N_crop, offset=short_axis) # plot right stack peaks
        plt.axvline(x=short_axis-1, color='black', linestyle='-', linewidth=0.5) # draw a line between the 2 stacks
        plt.title(f"{len(peaks)} Molecule{'s' if len(peaks) != 1 else ''} Detected")
        plt.imshow(img, cmap=parula,vmin=vmin,vmax=vmax); plt.tight_layout(); plt.colorbar(); plt.show()

    else : # single plane 
        conv_img = conv_img_left
        peaks = peaks_left
        print(sorted(tmp_sn_ratio_left, reverse=True))

        plt.figure(figsize=figsize)
        plt.subplot(1,2,1)
        plt.imshow(conv_img, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Convoluted Image'); plt.tight_layout(); plt.colorbar()
        draw_peaks(tmp_peaks_left, N_crop, offset=0, threshold_rel=threshold_rel, sn_ratio=tmp_sn_ratio_left)
        
        plt.subplot(1,2,2)
        draw_peaks(peaks, N_crop)
        plt.imshow(img, cmap=parula,vmin=vmin,vmax=vmax)
        plt.title('Image'); plt.tight_layout(); plt.colorbar()
        plt.suptitle(f"{len(peaks)} Molecule{'s' if len(peaks) != 1 else ''} Detected")
        plt.show()


  

#@partial(jax.jit, static_argnames=('sigma', 'min_dist', 'thresh', 'N_crop'))
def preloc_and_crop(stack, sigma=1, min_dist=15, threshold_abs=0, threshold_rel=1.5, N_crop=17, biplane=False, show_progress=True):
    ''' Performs the preloc: returns the preloc table and the cropped images of molecules'''

    # initials
    min_dist = max(min_dist, N_crop // 2 + 2)  # Minimum distance between peaks and image border
    long_axis = stack[0].shape[1]
    short_axis = long_axis // 2 if biplane else long_axis
    peaks, cropped_images = [],[]

    if show_progress: # print progress
        print('--- Preloc started ---')
        print('# frames:', '\t',len(stack))
        print('Analizing ... ')
        print(' 0 - 10  %')
        size, counter = len(stack) // 10, 10

    for m, img in enumerate(stack): # loop over frames

        if show_progress:
            if (m + 1) % size == 0:
                if counter < 90 :
                    counter += 10
                    print(counter - 10, '-', counter, ' %')
                elif counter == 90 :
                    print('90 - 100 %')
                    counter += 10

        # peaks left channel (if not biplane left channel = only channel)
        img_left = img[:, : short_axis]
        _, tmp_peaks_left = find_peaks(img_left, sigma, min_dist)

        if biplane :
            # peaks right channel
            img_right = img[:, short_axis:]
            _, tmp_peaks_right = find_peaks(img_right, sigma, min_dist)

            # merge  peaks
            peaks_left, _ , sn_ratio_left, _ = crop_images(img_left, tmp_peaks_left, N_crop, threshold_abs, threshold_rel)  
            peaks_right, _ , sn_ratio_right, _ = crop_images(img_right, tmp_peaks_right, N_crop, threshold_abs, threshold_rel)
            tmp_peaks = merge_double_peaks(peaks_left, peaks_right, sn_ratio_left, sn_ratio_right)
            
            #crop molecules from both channels
            if len(tmp_peaks) == 0 : continue # if there are no peaks in both stacks skip the frame
            _, cropped_images_left,_,_ = crop_images(img_left, tmp_peaks, N_crop)
            _, cropped_images_right,_,_ = crop_images(img_right, tmp_peaks, N_crop)
            tmp_cropped_images = np.concatenate((cropped_images_left, cropped_images_right), axis=2)

        else: # single plane
            tmp_peaks, tmp_cropped_images , _, _ = crop_images(img_left, tmp_peaks_left, N_crop, threshold_abs, threshold_rel)

        # frame, x coordinate (columns), y coordinate(rows)
        peaks.extend((m + 1, peak[1], peak[0]) for peak in tmp_peaks)  
        cropped_images.extend(tmp_cropped_images)

    data = np.hstack((np.arange(1,len(peaks)+1)[:, None], np.array(peaks)))  # Adding molecule index column
    table = pd.DataFrame(data, columns=["id", "frame", "x", "y"])
    print('--- Preloc completed ---') if show_progress else None
    print('# molecules: ', '\t', len(table))

    return table, np.array(cropped_images)



#----------------------------------------



#   FIT FUNCTIONS

#   - estimate_v0
#   - perform_fit
#   - lm_poisson
#   - compute_LLR
#   - fit_step
#   - add_preloc_positions




@partial(jax.jit, static_argnames=('h'))
def estimate_v0(images, h, z_start=None):
    '''initial fit guess '''  

    v0 = jnp.zeros((images.shape[0], 5))
    z0 = h.data.shape[2]//2 if z_start is None else z_start
    bg0 = images.min(axis=(1,2))
    s0 = (images - bg0[:,jnp.newaxis,jnp.newaxis]).sum(axis=(1,2))

    v0 = v0.at[:,:2].set(0)         # x0, y0 = 0
    v0 = v0.at[:,2].set(z0)         # z0 = (z_max-z_min) / 2, or z_start
    v0 = v0.at[:,3].set(s0)         # s0 = sum(I_photons - bg)
    v0 = v0.at[:,4].set(bg0)        # bg0 = min(I_photons)
    return v0



def perform_fit(I_photons, preloc_table, h, batch_size = 1000, z_start=None, factor_up = 7, factor_down=9, lmbda=1., maxiter = 100,  show_progress = False):
    ''' Performs the fit'''
    
    lm_poisson_vmap = jax.jit(jax.vmap(lambda x,v0,lmbda: lm_poisson(h, x, v0, tol=1e-1, maxiter=maxiter, lmbda=lmbda, factor_up=factor_up, factor_down=factor_down)))
    
    N_img = I_photons.shape[0]
    results_list, fit_results_list = [], []
    tot_iterations = N_img//batch_size
    iteration_counter = 0

    for i in range(tot_iterations+1): 

        if show_progress:
            iteration_counter +=1
            if iteration_counter == 1 :
                print("--- Precise fit ---")
                print('# of Batches :', tot_iterations+1)
                print('Progress:')
                print(iteration_counter) #, '\t/\t', tot_iterations+1)
            else: 
                print(iteration_counter) #, '\t/\t', tot_iterations+1)
    
        if i == (tot_iterations):
            sl_batch = slice(i*batch_size,None)
        else:
            sl_batch = slice(i*batch_size,(i+1)*batch_size)

        I_photons_jnp = jnp.array(I_photons[sl_batch])           # images
        v0_jnp = estimate_v0(I_photons_jnp, h) if z_start is None else estimate_v0(I_photons_jnp, h, z_start)                   # initial guess
        lmbda_vmap = jnp.ones(I_photons_jnp.shape[0]) * lmbda     # 
        est_precise_jnp, LLR_jnp = lm_poisson_vmap(I_photons_jnp, v0_jnp, lmbda_vmap)
        est_precise, LLR = np.asarray(est_precise_jnp),np.asarray(LLR_jnp)      
        results, fit_results = add_preloc_positions(est_precise, LLR, preloc_table[sl_batch], h)
        results_list.append(results), fit_results_list.append(fit_results)

    results = pd.concat(results_list)
    fit_results = pd.concat(fit_results_list)

    if show_progress: print("--- Fit completed ---")

    return results, fit_results



@partial(jax.jit, static_argnames=('h','maxiter'))
def lm_poisson(h, x, v0, tol=1e-3, maxiter=100, lmbda=1e-0, factor_up=7, factor_down=9):  
    """LM algorithm adapted to Poissonian MLE; see suppl. mat of :
    Laurence, Ted A., and Brett A. Chromy. "Efficient maximum likelihood estimator fitting of histograms." Nature methods 7.5 (2010): 338-339."""
    est_precise = v0.copy()

    LLR = jnp.inf
    def body_fun(i,val):
        
        est_precise,lmbda,LLR = val
        return fit_step(h, x, est_precise, tol, lmbda, factor_up, factor_down)
    
    est_precise, lmbda, LLR = jax.lax.fori_loop(0, maxiter, body_fun, (est_precise,lmbda,LLR))
    return est_precise, LLR


def compute_LLR(model, data) :
    """Computes image by image poissonian log likelihood ratio (LLR)"""

    positive_model = jnp.where(model <= 0, 1., model)    # replace log(0) = -inf with log(1) = 0
    positive_data = jnp.where(data <= 0, 1., data)       # replace log(0) = -inf with log(1) = 0
        
    if len(model.shape) == 3 : # stack of images
        return 2*jnp.sum((model - data * jnp.log(positive_model)) - (data - data * jnp.log(positive_data)), axis = (1, 2)) #log-likehood ratio for fit quality check
    elif len(model.shape) == 2 : # single image
        return 2*jnp.sum((model - data * jnp.log(positive_model)) - (data - data * jnp.log(positive_data))) #log-likehood ratio for fit quality check


def compute_least_squares(model, data):
    """calculates the least square error (Gaussian likelihood) for a given model and recorded data"""
    if len(model.shape) == 3 : # stack of images
        return jnp.sum(((model - data)**2) , axis = (1, 2))
    elif len(model.shape) == 2 : # single image
        return jnp.sum((model - data)**2)


#@partial(jax.jit, static_argnames=('h',))
def fit_step(h, x, a, tol, lmbda, factor_up, factor_down):
    ''' Fit: Levenberg-Marquardt algorithm'''
    ''' (J*J + lambda)delta = J*(x-f)'''
    ''' alpha = J*J + lambda'''
    ''' beta = J*(x-f)'''

    lmbda_min = 1e-5
    lmbda_max = 1e5
    n = len(a)

    f_a = h.f(a) # f evaluated at point a : f is the interpolation of the image with the PSF 
    J = jax.jacfwd(h.f)(a).reshape(-1,n) # Jacobian of f at point a
    
    tmp = x / jnp.square(f_a)
    JTJ = jnp.einsum('kj,kl->jl',J,J * tmp.flatten()[:,jnp.newaxis])
    JTr = -((1. - x / f_a).flatten().reshape(1,-1) @ J).flatten()  # gradient
    diag_JTJ = jnp.eye(n) * JTJ
    a_step  = solve(JTJ + lmbda * diag_JTJ, JTr) # calculated step. isresult of the linear system alpha * delta = beta
    f_a_new = h.f(a + a_step) # new function evaluation with updated parameters
    LLR = compute_LLR(f_a , x)
    LLR_new = compute_LLR(f_a_new , x)

    f_a_J = f_a.reshape(-1) + J @ a_step
    f_a_J_log = jnp.where(f_a_J <= 0, 1., f_a_J)
    
    f_a_log = jnp.where(f_a <= 0, 1., f_a).reshape(-1)
    rho_denom = 2. * (x.reshape(-1) * jnp.log(f_a_J_log) - x.reshape(-1) * jnp.log(f_a_log) - J @ a_step).sum()
    
    rho_denom = jnp.where(rho_denom == 0., jnp.inf, rho_denom)
    rho = (LLR - LLR_new) / rho_denom    
    cond = rho > tol
    P_lmbda = lambda l: jnp.clip(l, lmbda_min, lmbda_max)
    
    lmbda = jnp.where(cond, P_lmbda(lmbda / factor_down), P_lmbda(lmbda * factor_up))  # Decrease lambda if the model is closer to the minimum
                                                                                       # Increase lambda if the model is further from the minimum
    a = jnp.where(cond, a + a_step, a)          # do not update a

    # Clip the parameters. After the fit iterations are conlcuded.
    a = a.at[0:2].set(jnp.clip(a[0:2], -10*h.os, 10*h.os))      # clip x, y to +/- 10 pixels
    a = a.at[2].set(jnp.clip(a[2], 0, h.data.shape[2]))         # clip z at the PSF Z edges
    a = a.at[3:5].set(jnp.clip(a[3:5], 0.001 , jnp.inf))        # clip signal and bg to positive values

    LLR_ud = jnp.where(cond, LLR_new, LLR) # update the LLR
        
    return a, lmbda, LLR_ud



def add_preloc_positions(est_precise, LLR, preloc_table, h):
    """
    Adds prelocs positions to fit results.
    results: fit results with prelocs. space units are nanometers.
    raw_results: fit results without prelocs. space units are pixels (x,y) and PSF steps (z). Used for visualization and filtering.
    """
    results = pd.DataFrame(data = { 'id': preloc_table['id'],
                                    'frame': preloc_table['frame'], 
                                    'x': (est_precise[:, 1] + np.round(preloc_table["x"]*h.os)) * h.ux/h.os * 1e9, # nm  x: columns
                                    'y': (est_precise[:, 0] + np.round(preloc_table["y"]*h.os)) * h.ux/h.os * 1e9, # nm  y: rows
                                    'z': (est_precise[:, 2] * h.uz + h.z_min ) * 1e9, # nm
                                    'signal': est_precise[:, 3], # photon number
                                    'bg': est_precise[:, 4], # photon number
                                    'LLR': LLR})
    
    raw_results = pd.DataFrame(data = { 'id': preloc_table['id'], 
                                        'frame': preloc_table['frame'], 
                                        'x' : est_precise[:,1] / h.os, # pixels, x: columns
                                        'y' : est_precise[:,0] / h.os, # pixels, y: rows
                                        'z' : est_precise[:,2]}) # uz steps 

    return results, raw_results



#----------------------------------------









# DATA DATA AQCUISITION FUNCTIONS

#  - zernikes_decomposition
#  - z_drift_correction
#  - z_tilt_correction
#  - fit_signal
#  - find_structures
#  - find_single_emitters
#  - process_single_nanoruler
#  - statistics_nanorulers





from scipy.interpolate import BSpline, LSQUnivariateSpline, make_smoothing_spline


def zernikes_decomposition(phase, z_modes=None, pupil=None, plot=True, figsize=(14,4)):
    '''"
    Finds the Zernike coefficients of a phase image using a given set of Zernike modes.
    pupil: you can restrict the pupil where the zernikes are defined, for example to ignore SAF contribution. If None, the whole NA is considered.
    '''
    Nk = phase.shape[0]

    if pupil is None: # pupil is the whole NA
        pupil = zernike_noll(1, Nk)

    else:
        Nk = int(np.ceil(2*np.sqrt(np.sum(pupil) / np.pi))) # new pupil defined on a smaller NA
        pupil = zernike_noll(1, Nk)

    if z_modes is None: # by default: spherical decomposition
        z_modes = [1, 4, 11, 22, 37, 56]
    n_modes = len(z_modes)

    zernike_basis = np.zeros((n_modes, phase.shape[0], phase.shape[0]))
    
    for (n,z) in enumerate(z_modes):

        mode = zernike_noll(z, Nk) # zernike defined on given NA

        if Nk < phase.shape[0]: # pad to match sizes
            pad_left = (phase.shape[0] - Nk) // 2
            pad_right = (phase.shape[0] - Nk) - pad_left
            mode = np.pad(mode, ((pad_left, pad_right), (pad_left, pad_right)), mode='constant')
        
        zernike_basis[n, :, :] = mode

    # Flatten the 2D phase and Zernike modes into 1D arrays for computation
    pupil_phase_flat = phase.flatten()
    n_modes = zernike_basis.shape[0]
    amplitudes = np.zeros(n_modes)

    for i in range(n_modes): # Compute amplitude using the orthogonality property of a scalar product
        zernike_mode_flat = zernike_basis[i, :, :].flatten()
        numerator = np.dot(pupil_phase_flat, zernike_mode_flat)
        denominator = np.dot(zernike_mode_flat, zernike_mode_flat)
        amplitudes[i] = numerator / denominator if denominator != 0 else 0

    phase_fitted = np.sum(zernike_basis * amplitudes[:, np.newaxis, np.newaxis], axis = 0)

    if plot:
        phase_diff = phase - phase_fitted
        sum_pupil = np.sum(pupil > 0)
        rms_phase = np.sqrt(np.sum(np.square(phase)) / sum_pupil)
        rms_fit = np.sqrt(np.sum(np.square(phase_fitted)) / sum_pupil)
        rms_diff = np.sqrt(np.sum(np.square(phase_diff)) / sum_pupil)
        v_min, v_max = np.min(np.concatenate((phase, phase_fitted))), np.max(np.concatenate((phase, phase_fitted)))

        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.imshow(phase, cmap='hsv')#, vmin=v_min, vmax=v_max); 
        plt.title('Phase to fit'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_phase:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)
        plt.subplot(132)
        plt.imshow(phase_fitted, cmap='hsv')#, vmin=v_min, vmax=v_max); 
        plt.title('Phase fitted'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_fit:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)
        plt.subplot(133)
        plt.imshow(phase_diff, cmap='hsv')#, vmin=v_min, vmax=v_max); 
        plt.title('Phase difference'); plt.colorbar()
        plt.text(0.98, 0.98, f"RMS: {rms_diff:.2f} rad", ha='right', va='top', transform=plt.gca().transAxes)

        plt.show()

        a_max = np.max([np.abs(np.min(amplitudes)), np.max(amplitudes)])
        plt.figure(figsize=(10,3))
        plt.plot( amplitudes, '.'); plt.title('zernikes amplitudes / rad')
        plt.xticks(ticks=range(len(z_modes)), labels=z_modes)  # Add Zernike mode labels
        plt.ylim(-(a_max + 0.5), a_max + 0.5); plt.grid()
        plt.show()

    return amplitudes, phase_fitted, zernike_basis



def z_drift_correction(data, poly_order, exp_time=1,  plot = True):
    'Corrects the drift in z with a polynomial fit of order poly_order. 3 and 5 works good'
    'exp_time: exposure time in ms. If 1, the time is in frames.'

    # data
    time = np.array(data['frame'])
    z_points = np.array(data['z'])

    # fit
    poly_coeffs = np.polyfit(time, z_points, poly_order)
    poly_fit = np.poly1d(poly_coeffs)
    fit_z_points = poly_fit(time)
    corrected_z_points = z_points - fit_z_points + np.mean(z_points) # drift corrected points
    drift = fit_z_points[-1] - fit_z_points[0]

    # fit on drift corrected points to check the correction
    poly_coeffs_2 = np.polyfit(time, corrected_z_points, poly_order)
    poly_fit_2 = np.poly1d(poly_coeffs_2)
    fit_z_points_2 = poly_fit_2(time)
    
    if plot:
        time_units = exp_time if exp_time == 1 else exp_time * 1e-3 / 60. # frames or minutes
        time_axis = time * time_units
        unit = "frame" if time_units == 1 else "minutes"
        drift_text = f"Drift: {drift:.1f} nm"
        plt.scatter(time_axis , z_points, s=0.005, label='data before correction')
        plt.plot(time_axis, fit_z_points, '--', color='red', label='fit before correction')
        plt.plot(time_axis, fit_z_points_2, '--', color='black', label=' fit after correction')
        plt.xlabel(f'Time / {unit}')
        plt.ylabel('Z / nm')
        plt.legend(loc = 'upper right')
        plt.title('Z Drift Correction')
        plt.text(0.05, 0.92, drift_text, fontsize = 13, ha='left', va='top', transform=plt.gca().transAxes)
        plt.show()

    return corrected_z_points


def z_tilt_correction(data, plot = True):

    x,y,z = np.array(data['x']), np.array(data['y']), np.array(data['z'])

    XY = np.vstack((x, y, np.ones(len(x)))).T
    m, n, b = np.linalg.lstsq(XY, z, rcond=None)[0]
    z_fit = m * x + n * y + b 
    z = z - z_fit + np.mean(z)

    return z



def fit_signal(results, h):
    ''' Fits the signal vs z with an exponential decay and plots the results.'''
    '''Returns the predicted signal values to use for the CRLB calculation.'''

    def exp_decay(x, I0, decay_rate, offset):
        return I0 * np.exp(- decay_rate * x) + offset

    z = np.array(results['z'])
    points = np.array(results['signal'])
    x_axis = np.linspace(h.z_min * 1e9, h.z_max * 1e9, h.data.shape[2])

    lower_bounds = [1e2, 0, -1e3]
    upper_bounds = [1e5, 1e-1, 1e3]
    popt, _ = curve_fit(exp_decay, z.flatten(), points, bounds = (lower_bounds, upper_bounds), maxfev=10000)
    a, b, c = popt
    
    plt.plot(x_axis, exp_decay(x_axis, a, b, c), color='red')
    plt.scatter(z, points, s=0.01)
    plt.xlabel('Z / nm')
    plt.ylabel('Signal / photons')
    plt.title('Fit Signal vs Z')
    plt.show()

    return exp_decay(x_axis, a, b, c)




from sklearn.cluster import DBSCAN, OPTICS

def find_structures(data, method, eps, min_samples, label, n_bins, figsize=(10,10)):
    " Find nano-structures like nanorulers or NPCs with DBSCAN, OPTICS, DBSCAN3D or OPTICS3D"

    # Perform clustering on the (x, y) coordinates (alzo z for DBSCAN3D)
    points = data[:,1:3] #x-y positions

    print("Clustering with %s" % method)
    


    if method == "DBSCAN" :
        clust = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'DBSCAN3D' :
        points = data[:,1:4]
        clust = DBSCAN(eps=eps, min_samples=min_samples)
    # elif method == "OPTICS" :
    #     clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size)
    # elif method == "OPTICS3D" :
    #     clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size)
        

    clusters = clust.fit_predict(points)
    mask = (clusters == -1)
    dict_clusters = {}    

    #plot heatmap
    x,y = points[:,0:2].T
    size = np.ceil((max(x)-min(x))/1000)*n_bins
    bins = int(size/5)
    # print(size, bins)

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins, bins])
    vmax = np.max(hist)/5
    plt.figure(figsize=figsize)
    plt.imshow(hist.T, cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmax=vmax, origin='lower')
    plt.gca().set_xticks([]); plt.gca().set_yticks([])
   # Add squares around each cluster
    unique_clusters = np.unique(clusters)
    for i in unique_clusters:
        if i == -1: continue
        dict_clusters[i] = data[clusters == i] 
        cluster_points = points[clusters == i]
        min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
        min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        rect = plt.Rectangle((min_x, min_y), width, height, edgecolor='1', facecolor='none', linewidth=1)
        plt.gca().add_patch(rect)
    
        # Add cluster label
        if label == True :
            labels = f'{i}'
            label_x = min_x + width / 2
            label_y = max_y + height / 2
            plt.annotate(labels, xy=(label_x, label_y), xytext=(label_x, label_y), color="white", fontsize=12, ha="center", va="center")

    plt.gca().invert_yaxis()  # Invert y-axis
    plt.tight_layout()
    plt.show()

    # Count clusters and noise
    n_clusters = len(np.unique(clusters))-1
    print("--- %s NPCs found---" % n_clusters)
    noise = list(clusters).count(-1)   #localizations not in clusters
    noise = 100*noise/len(data)
    print("--- %s%% localizations discarded---" % round(noise,1))

    columns = ['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR']
    df = pd.DataFrame(data[~mask], columns=columns)
    
    return dict_clusters, df


from sklearn.cluster import KMeans



def find_single_emitters(dict_nanorulers, method, eps, max_eps, min_samples, xi, min_cluster_size, scaling_factor) :
    """
    Cluster all the given nanoruler into dense emitters with DBSCAN or OPTICS.
    """
    
    print("Clustering with %s" % method)
    n_nanorulers = len(dict_nanorulers)
    counter = 0
    dict_molecules = {}
    
    #cycle over nanorulers
    for nanoruler in range(n_nanorulers) :

        if (nanoruler+1) % 10 == 0 or nanoruler == n_nanorulers - 1: # print progress
            progress_percentage = (nanoruler + 1) / n_nanorulers * 100
            print(f'{progress_percentage:.2f} % complete: {nanoruler + 1} / {n_nanorulers}')

        points = dict_nanorulers[nanoruler][:,1:4] / scaling_factor #x-y-z coordinates
        min_samples = int(np.clip(points.shape[0] / min_samples, 2, points.shape[0]))

        if method == "DBSCAN" :
            clust = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "OPTICS" :
            clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size)
    
        clusters = clust.fit_predict(points)
        points = points * scaling_factor
        #noise = points[clusters == -1]
        unique_clusters = np.unique(clusters)
        
        #cluster nanorulers into 2 molecules
        if [0] and [1] in unique_clusters and [2] not in unique_clusters :
        
            for molecule in range(2): #2 molecules
                dict_molecules[counter] = dict_nanorulers[nanoruler][clusters == molecule]
                counter += 1

    data_molecules = np.concatenate([np.array(value) for value in dict_molecules.values()])

    ratio = 100 * counter / 2 / n_nanorulers
    print(round(ratio,1), '%')    
    
    return dict_molecules, data_molecules



def process_single_nanoruler(data, method, eps, max_eps, min_samples, xi, min_cluster_size, scaling_factor):
    """
    Clusters the given single nanoruler into density emitters.
    Plots the results.
    """
    
    # prepare plot
    plt.style.use('dark_background')  
    fs = 20
    fig = plt.figure(figsize=(26, 12))
    ax1 = fig.add_subplot(131, aspect='equal')  # XY Projection
    ax2 = fig.add_subplot(132, aspect='equal', sharex=ax1)  # XZ Projection
    ax3 = fig.add_subplot(133, aspect='equal')  # YZ Projection
    color = ['gold','green','red']

    print("Clustering with %s" % method)

    if method == "DBSCAN" :
        clust = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "OPTICS" :
        clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size)

    dict_clusters = {}
    points = data[:,1:4] / scaling_factor #scale z coordinates according to the scaling factor
    clusters = clust.fit_predict(points)
    points = points * scaling_factor # rescale back z coordinates
    noise = points[clusters == -1]
    mask = np.where(clusters == -1, True, False)
    unique_clusters = np.unique(clusters)
    print('# Points:', points.shape[0])

    #plot noise
    ax1.scatter(noise[:, 0], noise[:, 1], c=color[2])
    ax2.scatter(noise[:, 0], noise[:, 2], c=color[2])
    ax3.scatter(noise[:, 1], noise[:, 2], c=color[2])   

    if [0] and [1] not in unique_clusters or len(unique_clusters) > 3: # bad clusterization
        print('Number of clusters different than 2.')
        noise = points
        distance = np.nan
        theta = np.nan

    else:
        if 1 not in unique_clusters or 2 in unique_clusters :
            print(' -', len(unique_clusters) -1, 'Clusters found. Did not clusterized correctly.')

        for i in range(2): #2 clusters
            dict_clusters[i] = data[clusters == i] 

            cluster_points = points[clusters == i]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_radii = np.std(cluster_points, axis=0)
                
            # Plot the points in the cluster
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color[i]) #XY
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 2], c=color[i]) #XZ
            ax3.scatter(cluster_points[:, 1], cluster_points[:, 2], c=color[i]) #YZ
            
            # Plot the ellipses representing the std
            ellipse1 = patches.Ellipse(cluster_mean[:2], 2 * cluster_radii[0], 2 * cluster_radii[1], color='b', linewidth=4, fill=False)
            ellipse2 = patches.Ellipse(cluster_mean[[0, 2]], 2 * cluster_radii[0], 2 * cluster_radii[2], color='b', linewidth=4, fill=False)
            ellipse3 = patches.Ellipse(cluster_mean[1:], 2 * cluster_radii[1], 2 * cluster_radii[2], color='b', linewidth=4, fill=False)
            ax1.add_patch(ellipse1); ax2.add_patch(ellipse2); ax3.add_patch(ellipse3)
        
            #print std first cluster
            if i == 0:
                std_text_xy =  f'StdX yellow: {cluster_radii[0]:.1f} nm'
                std_text_xz =  f'StdY yellow: {cluster_radii[1]:.1f} nm'
                std_text_yz =  f'StdZ yellow: {cluster_radii[2]:.1f} nm'
                
                ax1.text(0.05, 0.95, f'{std_text_xy}', transform=ax1.transAxes, verticalalignment='top', fontsize=fs-4)
                ax2.text(0.05, 0.95, f'{std_text_xz}', transform=ax2.transAxes, verticalalignment='top', fontsize=fs-4)
                ax3.text(0.05, 0.95, f'{std_text_yz}', transform=ax3.transAxes, verticalalignment='top', fontsize=fs-4)
        
            if i == 1:
                # Calculate the distances between mean points
                prev_cluster_points = points[clusters == i - 1]
                prev_cluster_mean = np.mean(prev_cluster_points, axis=0)
                distance_xy = np.linalg.norm(cluster_mean[:2] - prev_cluster_mean[:2])
                distance_z = np.abs(cluster_mean[2] - prev_cluster_mean[2])
                distance = np.linalg.norm(prev_cluster_mean - cluster_mean)
                theta = np.arctan(distance_z / distance_xy) * 180 / np.pi
                
                # Plot the lines connecting the mean points
                ax1.plot([prev_cluster_mean[0], cluster_mean[0]], [prev_cluster_mean[1], cluster_mean[1]], c='b', linewidth=4)
                ax2.plot([prev_cluster_mean[0], cluster_mean[0]], [prev_cluster_mean[2], cluster_mean[2]], c='b', linewidth=4)
                ax3.plot([prev_cluster_mean[1], cluster_mean[1]], [prev_cluster_mean[2], cluster_mean[2]], c='b', linewidth=4)
                            
                #Annotate std
                std_text_xy =  f'StdX green: {cluster_radii[0]:.1f} nm'
                std_text_xz =  f'StdY green: {cluster_radii[1]:.1f} nm'
                std_text_yz =  f'StdZ green: {cluster_radii[2]:.1f} nm'
                ax1.text(0.05, 0.9, f'{std_text_xy}', transform=ax1.transAxes, verticalalignment='top', fontsize=fs-4)
                ax2.text(0.05, 0.9, f'{std_text_xz}', transform=ax2.transAxes, verticalalignment='top', fontsize=fs-4)
                ax3.text(0.05, 0.9, f'{std_text_yz}', transform=ax3.transAxes, verticalalignment='top', fontsize=fs-4)



    fig.suptitle(f'Distance: {distance:.2f}   Polar Angle: {theta:.2f}°', fontsize=fs)
    ax1.set_title('XY', fontsize=fs-2)
    ax2.set_title('XZ', fontsize=fs-2)
    ax3.set_title('YZ', fontsize=fs-2)
    
    # invert y position: y axis in ax1 and x axis in ax3
    ax1.invert_yaxis()    
    # ax3.invert_xaxis()    

    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.style.use('default')
    # Count noise
    noise = 100*len(noise)/len(data)
    print("--- %s%% localizations discarded---" % round(noise,1))
    
    return dict_clusters, data[~mask] 


def statistics_nanorulers(dict_nanorulers, dict_molecules, boundaries) :
    '''
    
    simpler filtering only changes the mean intensity and background of the molecules, 
    is performed after the computation of distances, positions and angles.

    '''
       
    n_nanorulers = len(dict_nanorulers)
    n_molecules = len(dict_molecules)
    mean_position = []     #of molecules
    std_position = []      #of molecules
    mean_intensity = []    #of molecules
    mean_bkg = []          #of molecules
    distance = []          #between pair of molecules of a nanoruler
    angle = []             #z inclination of nanorulers
    n_locs = []            #in a nanoruler
    
    for i in range(0, n_molecules, 2) :
        
        molecule1 = dict_molecules[i][:,2:5]
        molecule2 = dict_molecules[i+1][:,2:5]
        localizations = (len(molecule1) + len(molecule2))
        
        if boundaries[0][0] < localizations < boundaries[0][1]: #filter by number of localizations
        
            position1 = np.mean(molecule1, axis=0)    
            position2 = np.mean(molecule2, axis=0)    
            length = np.linalg.norm(position1 - position2)

            if boundaries[1][0] < length < boundaries[1][1]: #filter by lenght of nanoruler

                mean_position.append(position1)
                mean_position.append(position2)
                distance.append(length)
                std_position.append(np.std(molecule1, axis=0))
                std_position.append(np.std(molecule2, axis=0))

                distance_xy = np.linalg.norm(position1[:2] - position2[:2])
                distance_z = np.abs(position1[2] - position2[2])
                angle.append(np.arctan(distance_z / distance_xy) * 180 / np.pi)
                n_locs.append(localizations)
                
                mean_intensity.append(np.mean(dict_molecules[i][:,5]))
                mean_intensity.append(np.mean(dict_molecules[i+1][:,5]))
                mean_bkg.append(np.mean(dict_molecules[i][:,6]))
                mean_bkg.append(np.mean(dict_molecules[i+1][:,6]))         
                
    mean_molecules = np.zeros((len(mean_position),5))
    mean_molecules[:,0:3] = np.array(mean_position)
    mean_molecules[:,3] = np.array(mean_intensity)
    mean_molecules[:,4] = np.array(mean_bkg)    
    mean_distance = np.mean(distance)
    err_distance = (np.sum(np.abs(np.array(distance) - mean_distance))) / (mean_molecules.shape[0]/2)

    n_molecules = len(mean_molecules)


    fig, axes = plt.subplots(1, 5, figsize=(25, 7))
    axes[0].hist(n_locs, bins=10)
    axes[1].hist(distance, bins=10)
    axes[2].hist(angle, bins=10)
    axes[3].hist(mean_molecules[:, 2], bins=10)
    axes[4].scatter(mean_molecules[:, 2], mean_molecules[:, 3])

    titles = ['Localizations per nanoruler', 
              f'Length nanorulers: {mean_distance:.2f} +/- {err_distance:.2f}',
              'Z angle', 
              'Z position', 
              'Intensity vs Z position'
              ]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=20)
    fig.suptitle(f'Nanorulers found: {n_molecules // 2:.0f}/{n_nanorulers:.0f} ({100 * n_molecules // 2 / n_nanorulers:.0f}%)', fontsize=25)
    plt.tight_layout()
    plt.show()
    
    
    return mean_molecules, std_position, distance, angle







#----------------------------------------

# PLOT FUNCTIONS
# - show_prelocs
# - show_molecules
# - show_results_3d
# - show_results_2d
# - show_results_1d
# - show_results_time
# - show_results_hist
# - show_pixelation



def show_prelocs(I_photons, n=24, figsize=(10, 10)):
    """Show some random cropped molecules."""
    idx = np.random.randint(0, I_photons.shape[0], n)
    rows, cols, m = int(np.ceil(n / 6)), 6, 0
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            if m < n:
                ax[i, j].imshow(I_photons[idx[m], :, :], cmap=parula)
                ax[i, j].axis('off')
                m += 1
    fig.show()


def show_molecules(images, filter_indices, results, raw_results=None, title='id', n=16):
    """Randomly shows good molecules and filtered molecules."""

    if isinstance(title, int): # If title is an integer, use the corresponding column
        title = results.columns[title] if 0 <= title < len(results.columns) else results.columns[0]
    else:
        title = title if title in results.columns else results.columns[0]
    if n % 4 != 0: n = n + 4 - n % 4  # Make sure n is a multiple of 4

    # divide images and results table into 2 groups: good and filtered
    good_images, bad_images = images[filter_indices], images[~filter_indices]

    # generate random indices and select images to plot, with respective titles
    good_idx = np.random.choice(good_images.shape[0], n, replace=len(good_images) < n)
    bad_idx = np.random.choice(bad_images.shape[0], n, replace=len(bad_images) < n)
    good_images_plot, bad_images_plot = good_images[good_idx], bad_images[bad_idx]
    
    if title in ['x', 'y']:
        print('Header: (x, y) / pixels')
        good_results_x, bad_results_x = raw_results[filter_indices]['x'], raw_results[~filter_indices]['x']
        good_results_y, bad_results_y = raw_results[filter_indices]['y'], raw_results[~filter_indices]['y']
        good_results_1_plot, bad_results_1_plot = good_results_x.iloc[good_idx], bad_results_x.iloc[bad_idx]
        good_results_2_plot, bad_results_2_plot = good_results_y.iloc[good_idx], bad_results_y.iloc[bad_idx]

    elif title in ['signal', 'bg']:
        print('Header: (signal, bg) / photons')
        good_results_sg, bad_results_sg = results[filter_indices]['signal'], results[~filter_indices]['signal']
        good_results_bg, bad_results_bg = results[filter_indices]['bg'], results[~filter_indices]['bg']
        good_results_1_plot, bad_results_1_plot = good_results_sg.iloc[good_idx], bad_results_sg.iloc[bad_idx]
        good_results_2_plot, bad_results_2_plot = good_results_bg.iloc[good_idx], bad_results_bg.iloc[bad_idx]

    else:
        if title in ['id', 'frame', 'LLR']:
            print('Header:', title)
        else:
            print('Header:', title, '/ nm')

        good_results, bad_results = results[filter_indices][title], results[~filter_indices][title]
        good_results_plot, bad_results_plot = good_results.iloc[good_idx], bad_results.iloc[bad_idx]

    rows, cols, r, s = int(n / 4), 9, 0, 0
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.3 * rows))

    for i in range(rows):
        for j in range(4):  # First 4 columns for good molecules
            if r < n:
                axes[i, j].imshow(good_images_plot[r], cmap=parula)
                axes[i, j].axis('off')

                if title in ['x', 'y', 'signal', 'bg']:
                    axes[i, j].set_title(f"({good_results_1_plot.iloc[r]:.1f}, {good_results_2_plot.iloc[r]:.1f})", fontsize=8)
                else:
                    axes[i, j].set_title(f"{good_results_plot.iloc[r]:.0f}", fontsize=8)
                r += 1
        if j == 3:  # Skip the 5th column (index 4)
            axes[i, 4].axis('off')
    for i in range(rows):
        for j in range(5, 9):  # Last 4 columns for bad molecules
            if s < n:
                axes[i, j].imshow(bad_images_plot[s], cmap=parula)
                axes[i, j].axis('off')

                if title in ['x', 'y', 'signal', 'bg']:
                    axes[i, j].set_title(f"({bad_results_1_plot.iloc[s]:.1f}, {bad_results_2_plot.iloc[s]:.1f})", fontsize=8)
                else:
                    axes[i, j].set_title(f"{bad_results_plot.iloc[s]:.0f}", fontsize=8)                
                s += 1

    fig.suptitle(" ", fontsize=16)
    fig.text(0.23, 0.95, "Good Molecules", ha='center', fontsize=18)
    fig.text(0.777, 0.95, "Filtered Molecules", ha='center', fontsize=18)
    plt.tight_layout()
    plt.show()




def show_results_3d(data, marker_size=1.5): 
    ''' Show the 3d scatter plot with plotly'''

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        data = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        # x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']

    fig = plt.figure()
    fig = px.scatter_3d(data, x='x', y='y', z='z', color='z',
                        opacity=1, size_max=0.001, width=800, height=800)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=True), #False),
            yaxis=dict(showticklabels=True), #False),
            zaxis=dict(showticklabels=True), #False),
        ),
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Save as SVG",
                        method="restyle",  # No direct method; use custom script instead
                        args=[],
                        execute=True,
                        name="image"
                    )
                ],
            )
        ]
    )


    # fig.write_image("image.svg")
    fig.show()




def show_results_2d(data, marker_size=0.5):
    """Display 2D scatter plots for X/Y, X/Z, and Y/Z."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    plt.figure(figsize=(8, 5)); fs = 12

    def create_subplot(subplot_num, rowspan, colspan, title, x_data, y_data):
        ax = plt.subplot2grid((2, 3), subplot_num, rowspan=rowspan, colspan=colspan)
        ax.set_title(title, fontsize=fs)
        ax.scatter(x_data, y_data, s=marker_size)
        if subplot_num == (0, 0):
            ax.invert_yaxis()  # Invert y-axis for X/Y plot
            ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling for both axes

    create_subplot((0, 0), 2, 2, "X / Y",x, y)
    create_subplot((0, 2), 1, 1, "X / Z", x, z)
    create_subplot((1, 2), 1, 1, "Y / Z", y, z)

    plt.suptitle('Scatter Plot [nm]', fontsize=fs + 2)
    plt.tight_layout()
    plt.show()


def show_results_xy(data, show_labels=True, figsize=(10,10), marker_size=0.5):

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    width, height = figsize[0] * 100, figsize[1] * 100
    fig = px.scatter(x = x, y = y, color = z, opacity = 1)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(width=int(width), 
                    height=int(height),
                    xaxis=dict(
                        zeroline=False, 
                        showgrid=False, 
                        showticklabels=show_labels,
                        scaleanchor='y',  # Ensure same scale on both axes
                        scaleratio=1  # Ratio to be the same for both axes
                    ),
                    yaxis=dict(
                        zeroline=False,
                        autorange='reversed',
                        showgrid=False,
                        showticklabels=show_labels
                    ),
                    plot_bgcolor='black', 
                    paper_bgcolor='black'
                        )
    
    # fig.write_image("image.svg")
    fig.show()



def show_results_1d(data, raw_data=None, h=None, marker_size=0.5):
    """Display 1D plots for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    x_axis = np.arange(len(x))
    plt.figure(figsize=(8, 5)); fs = 12
    
    def create_subplot(subplot_num, title, data, x_axis, marker_size, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.plot(x_axis, data, '.', ms=marker_size)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)",    raw_data['x'],   x_axis, marker_size, fs)
        create_subplot(2, "Δy (pixels)",    raw_data['y'],   x_axis, marker_size, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)",    np.mod(x, h.ux * 1e9 / h.os),   x_axis, marker_size, fs)
        create_subplot(2, "mod(Δy) (pixels)",    np.mod(y, h.ux * 1e9 / h.os),   x_axis, marker_size, fs)

    create_subplot(3, "z (nm)",         z,      x_axis, marker_size, fs)
    create_subplot(4, "Photon number",  s,      x_axis, marker_size, fs)
    create_subplot(5, "Background",     bg,     x_axis, marker_size, fs)
    create_subplot(6, "LLR",            LLR,    x_axis, marker_size, fs)

    plt.suptitle('Localizations', fontsize=fs + 2)
    plt.tight_layout()
    plt.show()




def show_results_time(data, raw_data=None, h=None, exp_time = None, marker_size=0.5):
    """Display 1D plots vs frame of for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        frame, x, y, z, s, bg, LLR = data['frame'], data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        frame, x, y, z, s, bg, LLR = df['frame'], df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    time = frame * exp_time *1e-3/60 if exp_time else frame # time in minutes if exp_time is given, else time in frames

    plt.figure(figsize=(8, 5)); fs = 12
    
    def create_subplot(subplot_num, title, data, x_axis, marker_size, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.scatter(x_axis, data, s=marker_size)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)",    raw_data['x'],   time, marker_size, fs)
        create_subplot(2, "Δy (pixels)",    raw_data['y'],   time, marker_size, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)",    np.mod(x, h.ux * 1e9 / h.os),   time, marker_size, fs)
        create_subplot(2, "mod(Δy) (pixels)",    np.mod(y, h.ux * 1e9 / h.os),   time, marker_size, fs)

    create_subplot(3, "z (nm)",         z,      time, marker_size, fs)
    create_subplot(4, "Photon number",  s,      time, marker_size, fs)
    create_subplot(5, "Background",     bg,     time, marker_size, fs)
    create_subplot(6, "LLR",            LLR,    time, marker_size, fs)

    plt.suptitle('Localizations vs time (minutes)' if exp_time else 'Localizations vs frames', fontsize=fs + 2)

    plt.tight_layout()
    plt.show()




def show_results_hist(data, raw_data=None, h=None, bins=20):
    """Display histograms for Δx, Δy, z, Photon number, Background, and LLR."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        x, y, z, s, bg, LLR = data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        x, y, z, s, bg, LLR = df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']

    plt.figure(figsize=(8, 5));  fs = 12

    def create_subplot(subplot_num, title, data, bins, fs):
        plt.subplot(2, 3, subplot_num)
        plt.title(title, fontsize=fs)
        plt.hist(data, bins)
        plt.text(0.95, 0.95, f"μ: {np.mean(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.text(0.95, 0.85, f"σ: {np.std(data):.2f}", ha='right', va='top', transform=plt.gca().transAxes)

    if raw_data is not None:
        create_subplot(1, "Δx (pixels)", raw_data['x'], bins, fs)
        create_subplot(2, "Δy (pixels)", raw_data['y'], bins, fs)
    else : 
        create_subplot(1, "mod(Δx) (pixels)", np.mod(x, h.ux * 1e9 / h.os), bins, fs)
        create_subplot(2, "mod(Δy) (pixels)", np.mod(y, h.ux * 1e9 / h.os), bins, fs)

    create_subplot(3, "z (nm)",         z,      bins, fs)
    create_subplot(4, "Photon number",  s,      bins, fs)
    create_subplot(5, "Background",     bg,     bins, fs)
    create_subplot(6, "LLR",            LLR,    bins, fs)

    plt.suptitle('Histograms', fontsize=fs+3)
    plt.tight_layout()
    plt.show()




def show_pixelation(h, results, bins_plotted=1, bins=100):
    
    plt.figure(figsize=(10, 4))
    mod_x = np.mod(results['x'], bins_plotted * h.ux * 1e9 / h.os)
    mod_y = np.mod(results['y'], bins_plotted * h.ux * 1e9 / h.os)
    mod_z = np.mod(results['z'], bins_plotted * h.uz * 1e9)

    for i, (data, axis) in enumerate(zip([mod_x, mod_y, mod_z], ['X', 'Y', 'Z']), 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=bins)
        plt.title(axis)

    plt.suptitle('Pixelation', fontsize=15)
    plt.show()






#----------------------------------------

# FILTER FUNCTIONS
# - gaussian
# - dash_plot
# - dash_filtering
# - filter_results


def gaussian(x, offset, a, x0, sigma):
    return offset + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def dash_plot(data):
    ''' Creates an interactive plot with dash'''
    '''   Functionalities: '''
    ''' - Filter based on z and signal range'''
    ''' - Zoom in and out and plots will update accordingly'''
    ''' - Change z bin size and marker size'''
    ''' - Fit gaussians to the data'''

    # slider values for the z range
    z_min, z_max = np.floor(data['z'].min() / 50) * 50, np.ceil(data['z'].max() / 50) * 50
    z_values = np.arange(z_min, z_max + 200, 200).astype(int)

    # slider values for the signal range
    signal_min, signal_max = np.floor(data['signal'].min() / 1000) * 1000, np.ceil(data['signal'].max() / 1000) * 1000
    signal_values = np.arange(signal_min, signal_max + 500, (signal_max - signal_min) // 5).astype(int)


    app = dash.Dash(__name__)
    app.layout = html.Div([

        # plots
        html.Div([
            dcc.Graph(id='plot', config={'modeBarButtonsToAdd': ['drawrect', 'eraseshape' , 'lasso2d']}),
        ], style={'width': '80%', 'float': 'left'}),


        html.Div([


            # buttons
            html.Button('Fit Gaussians', id='fit-gaussians-button', n_clicks=0),
            html.Button('Change plot type', id='toggle-plot-button', n_clicks=0),
            html.Button('Save as SVG', id='save-svg-button', n_clicks=0),


            # sliders

            html.Label('  Marker Size', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='marker-size-slider',
                min=0.2,
                max=8,
                value=1,
                marks={i: str(i) for i in range(0, 9, 2)},
                step=0.2
            ),

            html.Label('  Signal Range (photons)', style={'font-weight': 'bold', 'color': 'white'}),
            dcc.RangeSlider(
                id='signal-range-slider',
                min=signal_min,
                max=signal_max,
                value=[signal_min, signal_max],
                marks={str(s): str(s) for s in signal_values},
                step=1
            ),

            html.Label('  Z Range (nm)', style={'font-weight': 'bold', 'color': 'white'}),
            dcc.RangeSlider(
                id='z-range-slider',
                min=z_min,
                max=z_max,
                value=[z_min, z_max],
                marks={str(z): str(z) for z in z_values},
                step=1
            ),

            html.Label('  XY bins', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='xy-bins-slider',
                min=10,
                max=100,
                value=25,
                marks={i: str(i) for i in range(10, 110, 10)},
                step=1
            ),

            html.Label('  Z bins', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='z-bins-slider',
                min=10,
                max=100,
                value=25,
                marks={i: str(i) for i in range(10, 110, 10)},
                step=1
            ),

            html.Label('  Fit initials: Gaussians Z center of mass (nm)', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='initial-center-slider',
                min=-50,
                max=50,
                value=0,
                marks={i: str(i) for i in range(-60, 60, 10)},
                step=1
            ),

            html.Label('  Fit initials: Gaussians distance (nm)', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='initial-distance-slider',
                min=20,
                max=180,
                value=50,
                marks={i: str(i) for i in range(20, 200, 20)},
                step=1
            ),

            html.Label('  Fit initials: Gaussians sigma 1 (nm)', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='initial-sigma1-slider',
                min=1,
                max=40,
                value=10,
                marks={i: str(i) for i in range(0, 45, 5)},
                step=0.1
            ),

            html.Label('  Fit initials: Gaussians sigma 2 (nm)', style={'font-weight': 'bold', 'color': 'white', 'margin-top': '20px'}),
            dcc.Slider(
                id='initial-sigma2-slider',
                min=1,
                max=40,
                value=10,
                marks={i: str(i) for i in range(0, 45, 5)},
                step=0.1
            ),

            dcc.Store(id='original-data', data=data.to_dict('records')),
            dcc.Store(id='zoomed-data', data=data.to_dict('records')),

            dcc.Download(id='download-svg'),

        ], style={'width': '20%', 'float': 'right'})
    ])

    @app.callback(
        Output('plot', 'figure'),
        Output('zoomed-data', 'data'),
        
        Input('plot', 'relayoutData'),
        Input('plot', 'selectedData'),

        Input('marker-size-slider', 'value'),
        Input('signal-range-slider', 'value'),
        Input('z-range-slider', 'value'),

        Input('xy-bins-slider', 'value'),
        Input('z-bins-slider', 'value'),


        Input('initial-center-slider', 'value'),
        Input('initial-distance-slider', 'value'),
        Input('initial-sigma1-slider', 'value'),
        Input('initial-sigma2-slider', 'value'),

        Input('toggle-plot-button', 'n_clicks'),
        Input('fit-gaussians-button', 'n_clicks'),

        State('original-data', 'data'),
        State('zoomed-data', 'data'),
        prevent_initial_call=True
    )


    def update_figure(relayout_data, selected_data, marker_size, signal_range, z_range, xy_bins, z_bins, init_z_center, init_dist, init_sigma1, init_sigma2, toggle_n_clicks, fit_n_clicks, original_data, zoomed_data):

        global df
        df = pd.DataFrame(original_data)

        # Filter data based on sliders
        df = df[(df['z'] >= z_range[0]) & (df['z'] <= z_range[1])]
        df = df[(df['signal'] >= signal_range[0]) & (df['signal'] <= signal_range[1])]

        # Update zoomed data based on selection or relayoutData
        if selected_data and 'points' in selected_data:
            selected_points = [point['pointIndex'] for point in selected_data['points']]
            # selected_indices = [point['pointIndex'] for point in selected_points]
            selected_indices = selected_points
            df = pd.DataFrame(zoomed_data).iloc[selected_indices]




        elif relayout_data: # zoomed data based on relayoutData

            # zoomed data from scatter plot XY (Y axis is reversed)
            xy_x0, xy_x1 = relayout_data.get('xaxis.range[0]', min(data['x'])), relayout_data.get('xaxis.range[1]', max(data['x']))
            xy_y0, xy_y1 = relayout_data.get('yaxis.range[1]', min(data['y'])), relayout_data.get('yaxis.range[0]', max(data['y']))

            # zoomed data from scatter plot XZ
            xz_z0, xz_z1 = relayout_data.get('yaxis2.range[0]', min(data['z'])), relayout_data.get('yaxis2.range[1]', max(data['z']))
            xz_x0, xz_x1 = relayout_data.get('xaxis2.range[0]', min(data['x'])), relayout_data.get('xaxis2.range[1]', max(data['x']))

            # zoomed data from scatter plot YZ
            yz_y0, yz_y1 = relayout_data.get('xaxis4.range[0]', min(data['y'])), relayout_data.get('xaxis4.range[1]', max(data['y']))
            yz_z0, yz_z1 = relayout_data.get('yaxis4.range[0]', min(data['z'])), relayout_data.get('yaxis4.range[1]', max(data['z']))

            df = df[(df['x'] >= xy_x0) & (df['x'] <= xy_x1) & (df['y'] >= xy_y0) & (df['y'] <= xy_y1) & 
                    (df['z'] >= xz_z0) & (df['z'] <= xz_z1) & (df['x'] >= xz_x0) & (df['x'] <= xz_x1) &
                    (df['y'] >= yz_y0) & (df['y'] <= yz_y1) & (df['z'] >= yz_z0) & (df['z'] <= yz_z1)]

        fig = make_subplots(
            rows=2, cols=4,
            shared_xaxes=False,
            specs=[
                [{"rowspan": 2, "colspan": 2}, None, {}, {}],   # first row
                [None, None, {}, {}]],                          # second row
            subplot_titles=("XY", 'XZ',  "Z Histogram", "YZ", 'Photons Histogram'))

        if toggle_n_clicks % 2 == 0:  # Scatter plot

            scatter_xy = go.Scattergl( # XY
                x=df['x'], y=df['y'], 
                mode='markers', 
                marker=dict(color=df['z'], size=marker_size, colorscale='plasma'),
                name='XY Scatter')
            fig.add_trace(scatter_xy , row=1, col=1)
            fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) # This ensures the x and y axes have the same scale

            scatter_xz = go.Scattergl( # XZ
            x=df['x'], y=df['z'], 
            mode='markers', 
            marker=dict(color=df['z'], size=marker_size, colorscale='plasma'),
            name='XZ Scatter')
            fig.add_trace(scatter_xz , row=1, col=3)

            scatter_yz = go.Scattergl( # YZ
            x=df['y'], y=df['z'], 
            mode='markers', 
            marker=dict(color=df['z'], size=marker_size, colorscale='plasma'),
            name='YZ Scatter')
            fig.add_trace(scatter_yz , row=2, col=3)

        else:  # Heatmap
            heatmap_xy = go.Histogram2d( # XY Heatmap
                x=df['x'], y=df['y'],
                nbinsx=xy_bins, nbinsy=xy_bins,
                colorscale='hot',
                showscale=False,
                name='XY Heatmap')
            fig.add_trace(heatmap_xy, row=1, col=1)
            fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) # This ensures the x and y axes have the same scale

            heatmap_xz = go.Histogram2d(  # XZ Heatmap
                x=df['x'], y=df['z'],
                nbinsx=xy_bins, nbinsy=z_bins,
                colorscale='hot',
                showscale=False,
                name='XZ Heatmap')
            fig.add_trace(heatmap_xz, row=1, col=3)
            
            heatmap_yz = go.Histogram2d( # YZ Heatmap
                x=df['y'], y=df['z'],
                nbinsx=xy_bins, nbinsy=z_bins,
                colorscale='hot',
                showscale=False,
                name='XY Heatmap')
            fig.add_trace(heatmap_yz, row=2, col=3)



        # Z Histogram and Gaussian Fit
        hist_data = df['z']
        x_min, x_max = int(np.floor(hist_data.min())), int(np.ceil(hist_data.max())) # edges of hinstogram
        hist_values, bin_edges = np.histogram(hist_data, bins=np.linspace(x_min, x_max, z_bins+1), range=(x_min, x_max))#, density=True)
        hist_max = np.max(hist_values)
        hist_z = go.Histogram(
                x=hist_data,
                xbins=dict(start=x_min, end=x_max, size=((x_max-x_min)/z_bins)),  # Matching the bins
                name='Z Histogram',
                marker_line_color='black', marker_line_width=1,
                marker=dict(color=px.colors.qualitative.Plotly[0]))
        fig.add_trace(hist_z, row=1, col=4)
        fig.update_layout(yaxis3=dict(
                tickvals=[v for v in np.linspace(0, hist_max, 3)],
                ticktext=[f"{v / hist_max:.1f}" for v in np.linspace(0, hist_max, 3)]))

        if fit_n_clicks % 2 != 0: # compute and plot gaussians
            try:
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                offset_guess = 0.1
                a1_guess, a2_guess = hist_max, hist_max
                x01_guess, x02_guess = np.mean(hist_data) + init_z_center - init_dist / 2, np.mean(hist_data) + init_z_center + init_dist / 2
                sigma1_guess, sigma2_guess = init_sigma1, init_sigma2
                popt, _ = curve_fit(lambda x, offset, a1, x01, sigma1, a2, x02, sigma2: gaussian(x, offset, a1, x01, sigma1) + gaussian(x, offset, a2, x02, sigma2), # fit function
                                    bin_centers, hist_values, # x, y data to fit
                                    p0 = [offset_guess, a1_guess, x01_guess, sigma1_guess, a2_guess, x02_guess, sigma2_guess], # initial guess
                                    bounds = ([0, a1_guess / 4 ,   -np.inf, sigma1_guess / 1.5,      a2_guess / 4,   -np.inf, sigma2_guess / 1.5],  # lower bounds
                                              [0.5, a1_guess * 1.2,   np.inf, sigma1_guess * 1.5,      a2_guess * 1.2,  np.inf, sigma2_guess * 1.5])) # upper bounds
                offset, a1, x01, sigma1, a2, x02, sigma2 = popt
                distance = np.abs(x02 - x01)
                print(f'Peaks Distance:  {distance:.1f} nm', '\t', 
                      f'Peaks Centers: {x01:.1f} / {x02:.1f} nm', '\t', 
                      f'Sigma: {sigma1:.1f} / {sigma2:.1f} nm', '\t', 
                      f'Amp: {a1/hist_max:.2f} / {a2/hist_max:.2f}', '\t',
                      f'Offset: {offset:.2f}')

                x_axis = np.linspace(hist_data.min() - 50, hist_data.max() + 50, 200)
                fit_curve = gaussian(x_axis, offset, a1, x01, sigma1) + gaussian(x_axis, offset, a2, x02, sigma2)
                fig.add_trace(go.Scatter(x=x_axis, y=fit_curve, mode='lines', name='Fit', line=dict(color='red')), row=1, col=4)
                fig.add_shape(type="line", x0=x01, y0=0, x1=x01, y1=gaussian(x01, offset, a1, x01, sigma1), line=dict(color="red", dash="dash"), row=1, col=4)
                fig.add_shape(type="line", x0=x02, y0=0, x1=x02, y1=gaussian(x02, offset, a2, x02, sigma2), line=dict(color="red", dash="dash"), row=1, col=4)
                fig.add_annotation( # print peaks distance
                    x = (x_min + x_max) / 2, y = hist_max * 1.1,  # slightly above the histogram for visibility
                    text=f'Distance: {distance:.2f} nm', showarrow=False, font=dict(size=14, color="red"), row=1, col=4)
            except Exception as e:
                print(f"Gaussian fit failed: {e}")


        # Hist signal
        hist_signal = go.Histogram(
            x=df['signal'],
            nbinsx=50,
            name='Photon number Histogram',
                marker_line_color='black',  # Black edges
                marker_line_width=1,         # Width of the edge
                marker=dict(color=px.colors.qualitative.Plotly[0]))  # Lighter blue color
        fig.add_trace(hist_signal, row=2, col=4)

        # Final layout for all plots
        fig.update_layout( 
            width=1400, 
            height=700,
            showlegend=False,
            xaxis_title="X", yaxis_title="Y", yaxis=dict(autorange='reversed'),
            xaxis2_title="X", yaxis2_title="Z",
            xaxis3_title="Z", 
            xaxis4_title="Y", yaxis4_title="Z",
            xaxis5_title="Photon number",
        )

        return fig, df.to_dict('records')
    
    @app.callback(
        Output('download-svg', 'data'),
        Input('save-svg-button', 'n_clicks'),
        State('plot', 'figure'),
        prevent_initial_call=True
    )
    def export_svg(n_clicks, figure):
        if n_clicks > 0:
            img_bytes = BytesIO()
            fig = go.Figure(figure)
            fig.write_image(img_bytes, format='svg')
            img_bytes.seek(0)
            return dcc.send_bytes(img_bytes.getvalue(), filename="plot.svg")

    return app







    
def dash_filtering(results):
    # Global variable to store the selected DataFrame
    global selected_df, selected_indices
    selected_df = pd.DataFrame()
    selected_indices = [False] * len(results)

    # Create the Dash app
    app = dash.Dash(__name__)

    # Define the marker sizes
    marker_sizes = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5]

    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='scatter-plot',
                config={'modeBarButtonsToAdd': ['lasso2d']},
                style={'width': '70%', 'display': 'inline-block', 'float': 'left'}
            ),
            html.Div([
                html.Label("Select Scatter Plot:"),
                dcc.Dropdown(
                    id='scatter-type',
                    options=[
                        {'label': 'xy', 'value': 'xy'},
                        {'label': 'xz', 'value': 'xz'},
                        {'label': 'yz', 'value': 'yz'}
                    ],
                    value='xy'
                ),
                html.Label("Select Marker Size:"),
                dcc.Dropdown(
                    id='marker-size',
                    options=[{'label': str(size), 'value': size} for size in marker_sizes],
                    value=1
                ),
                html.Div(id='selected-data')
            ], style={'width': '30%', 'display': 'inline-block', 'float': 'right'})
        ]),
    ])

    # Callback to save the selected data as a DataFrame
    @app.callback(
        Output('selected-data', 'children'),
        Input('scatter-plot', 'selectedData'),
        State('scatter-plot', 'figure')
    )
    def display_selected_data(selectedData, figure):
        
        global selected_df, selected_indices

        if selectedData is None or not selectedData['points']:
            return "No data selected"

        selected_points = selectedData['points']
        selected_indices = [False] * len(results)
        
        for point in selected_points:
            selected_indices[point['pointIndex']] = True

        selected_df = results[selected_indices]

        # selected_indices = [point['pointIndex'] for point in selected_points]
        # selected_df = results.iloc[selected_indices]

        print('Points selected:', selected_df.shape[0])
        return "Selected data has been saved to the DataFrame."

    # Callback to update the scatter plot
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('scatter-type', 'value'),
        Input('marker-size', 'value'),
        State('scatter-plot', 'selectedData')
    )
    def update_scatter_plot(scatter_type, marker_size, selectedData):
        if scatter_type == 'xy':
            fig = px.scatter(results, 
                            x='x', 
                            y='y', 
                            opacity=1, 
                            color_discrete_sequence=['blue'])
            fig.update_layout(yaxis={'autorange': 'reversed'})

        elif scatter_type == 'xz':
            fig = px.scatter(results, 
                            x='x', 
                            y='z', 
                            opacity=1,
                            color_discrete_sequence=['blue'])

        elif scatter_type == 'yz':
            fig = px.scatter(results, 
                            x='y', 
                            y='z', 
                            opacity=1, 
                            color_discrete_sequence=['blue'])

        fig.update_layout(width=700, height=700)

        if selectedData and selectedData['points']:
            selected_points = selectedData['points']
            selected_indices = [point['pointIndex'] for point in selected_points]

            selected_fig = px.scatter(results.iloc[selected_indices], 
                                    x=fig.data[0]['x'],
                                    y=fig.data[0]['y'],
                                    size='z', 
                                    opacity=1,
                                    color_discrete_sequence=['blue']).data[0]

            fig.add_trace(selected_fig)

        fig.update_traces(marker=dict(size=marker_size))
        return fig

    return app


def filter_results(data, raw_data=None,
                   x_range=(-np.inf, np.inf), y_range=(-np.inf, np.inf), z_range=(-np.inf, np.inf),
                   signal_range=(1, 1e9), bg_range=(1, 1e3), LLR_range=(1, 1e9),
                   x_pxl=10, y_pxl=10, print_fraction=True):
    """Filter the results DataFrame based on the given ranges."""

    if isinstance(data, pd.DataFrame):  # for pandas DataFrame
        frame, x, y, z, s, bg, LLR = data['frame'], data['x'], data['y'], data['z'], data['signal'], data['bg'], data['LLR']
    elif isinstance(data, (np.ndarray, jnp.ndarray)):  # for NumPy array
        df = pd.DataFrame(data, columns=['frame', 'x', 'y', 'z', 'signal', 'bg', 'LLR'])
        frame, x, y, z, s, bg, LLR = df['frame'], df['x'], df['y'], df['z'], df['signal'], df['bg'], df['LLR']


    # Initialize indices as a boolean array with the same length as results
    indices = np.ones(len(data), dtype=bool)

    if raw_data is None:  # Filtering without pixel units when raw_results is not provided
        indices &= (
            x.between(x_range[0], x_range[1]) &
            y.between(y_range[0], y_range[1]) &
            z.between(z_range[0], z_range[1]) &
            s.between(signal_range[0], signal_range[1]) &
            bg.between(bg_range[0], bg_range[1]) &
            LLR.between(LLR_range[0], LLR_range[1])
        )
    else:  # Filtering with pixel units
        min_x, max_x = np.min(raw_data['x']), np.max(raw_data['x']) # Pre-filtering for pixelation artifacts
        prefilter_indices = (raw_data['x'].between(min_x + 0.1, max_x - 0.1) &
                             raw_data['y'].between(min_x + 0.1, max_x - 0.1))
        prefilt_raw = raw_data[prefilter_indices] # filter out the very edges of the fitted molecules
        mean_x, mean_y = np.mean(prefilt_raw['x']), np.mean(prefilt_raw['y'])

        indices &= prefilter_indices  # Apply the prefilter indices to the main indices
        indices &= (
            x.between(x_range[0], x_range[1]) &
            y.between(y_range[0], y_range[1]) &
            z.between(z_range[0], z_range[1]) &
            s.between(signal_range[0], signal_range[1]) &
            bg.between(bg_range[0], bg_range[1]) &
            LLR.between(LLR_range[0], LLR_range[1]) &
            raw_data['x'].between(mean_x - x_pxl, mean_x + x_pxl) &
            raw_data['y'].between(mean_y - y_pxl, mean_y + y_pxl)
        )
        filt_raw = raw_data[indices]
    filt_data = data[indices]

    if print_fraction:  # Fraction in %
        frac_filt = np.round((1 - len(filt_data) / len(data)) * 100)
        print('Localizations:', len(filt_data), '/', len(data), '\t |', frac_filt, '% filtered out.')

    if raw_data is None:
        return filt_data
    else:
        return filt_data, filt_raw, indices





#------------------------------------------------
#           GENERATE SYNTHETIC DATA             #
#------------------------------------------------


# - show_aberr
# - plot_aberr
# - generate_random_aberr
# - generate_spiral
# - generate_npc
# - generate_image_photons



def show_aberr(noll_index, aberr, aberr2=None, figsize=(5,3), title='Aberrations'):

    max_value = np.max(np.abs(aberr).max()) + 0.2
    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(aberr)), aberr, 'o')
    if aberr2 is not None:
        plt.plot(np.arange(len(aberr2)), aberr2, 'o')
    plt.xticks(np.arange(len(aberr)), np.int_(noll_index))
    plt.xlabel('Zernike mode (Noll index)')
    plt.ylabel('Magnitude (rad)')
    plt.title(f'{title}')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylim(-max_value, max_value)
    plt.grid()
    plt.show()



def plot_aberr(noll_index, aberr1, aberr2, figsize = (10,3)) :
    "Plot the results from the aberr retrieval"

    max_value = np.max([np.abs(aberr1).max(), np.abs(aberr2).max()]) + 0.3

    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(aberr1)), aberr1, 'o', label='Bead Aberrations')
    plt.plot(np.arange(len(aberr1)), aberr2, 'o', label='Deconvolved Aberrations')
    plt.xticks(np.arange(len(aberr1)), np.int_(noll_index))
    plt.xlabel('Zernike mode (Noll index)')
    plt.ylabel('Magnitude (rad)')
    plt.title(f'Aberrations')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylim(-max_value, max_value)
    plt.grid()
    plt.legend()
    plt.show()


def generate_random_aberr(noll_index, norm=0.5, psf='cyl_psf'):
    """ Genrate a random aberration vector with the given Noll indices """
    """ Radial order:"""
    """  0th:  1 """
    """  1st:  2, 3 """
    """  2nd:  4, 5, 6 """
    """  3rd:  7, 8, 9,10 """
    """  4th: 11,12,13,14,15 """
    """  5th: 16,17,18,19,20,21 """
    """  6th: 22,23,24,25,26,27,28 """
    """  7th: 29,30,31,32,33,34,35,36 """
    """  8th: 37,38,39,40,41,42,43,44,45 """
    """  9th: 46,47,48,49,50,51,52,53,54,55 """
    """ 10th: 56,57,58,59,60,61,62,63,64,65,66 """

    '''Spherical aberrations: 11, 22, 37, 56. Only on even Zernike orders'''
   

    rnd_aberr = np.random.uniform(-0.5, 0.5, len(noll_index))

    # make higher order aberrations less important: relative amplitude = c ^ (n-1), with n being the zernike radial order
    thresholds = [4, 7, 11, 16, 22, 29, 37, 46, 56]
    c = 0.85
    for threshold in thresholds:
        rnd_aberr[np.asarray(noll_index) >= threshold] *= c

    rnd_aberr *= norm / np.linalg.norm(rnd_aberr) # normalize to given norm

    # add a cyl mode
    if psf == 'cyl_psf':
        print('Aberrations norm before adding cyl mode:', np.round(np.linalg.norm(rnd_aberr), 3))
        cyl2 = np.where(np.asarray(noll_index) == 6)[0][0] # cyl2 index
        rnd_aberr[cyl2] *= 0.2
        rnd_aberr[cyl2] += 1.0
        print('Aberrations norm after adding cyl mode:', np.round(np.linalg.norm(rnd_aberr), 3))
    
    return rnd_aberr



def generate_spiral(h, cam,
                    s = 2000,
                    bg = 100,
                    N_mol = 8,
                    N_img = 1000,
                    z0 = 0e-9,
                    dz = 10e-9,
                    R = 80e-9,
                    on_fraction=0.1,
                    biplane = False,
                    z_biplane = 400):
    
    """
    Generate syntethic SMLM image stack of a spiral of molecules.
    if biplane = True, two stacks are generated with a z_shift between them.
    """
    I_sum = np.zeros((N_img, h.Nx, h.Nx)) # images are added to this stack
    N_on = int(round(N_img*on_fraction)) #number of images with the molecule on
    I_on = np.zeros((N_on, h.Nx, h.Nx)) # images with molecules

    if biplane:
        I_sum_2 = I_sum.copy()
        I_on_2 = I_on.copy()
        s /= 2.

        # generate the second PSF defocused by z_shift respect to the first one
        h2 = psf(z_min=h.z_min * 1e9,
                 z_max=h.z_max * 1e9,
                 z_defocus=h.z_defocus * 1e9 + z_biplane,
                 wavelength=h.wavelength * 1e9,
                 NA=h.NA,
                 Nx=h.Nx,
                 ux=h.ux * 1e9,
                 focal_len=h.focal_len * 1e9,
                 RI=h.RI,
                 uz=h.uz * 1e9,
                 Z_modes=h.Z_modes,
                 Z_magn=h.Z_magn,
                 os=h.os)


    for m in range(1, N_mol+1): # iterate over the molecules

        Image = np.zeros((N_img, h.Nx, h.Nx))
        idx = np.random.choice(N_img, N_on, replace=False) #randomly choose N_on indices

        # x, y, z positions
        x = R*np.cos(m*2*np.pi/N_mol) #molecule positions are distributed on a circle
        y = R*np.sin(m*2*np.pi/N_mol)
        z = z0 + (m-1)*dz
        
        I_on = generate_images_photons(h, [x, y, z, s, 0], N_img = N_on) # generate the image
        Image[idx,:,:] = I_on
        I_sum += Image #add to the other molecule images

        if biplane : # repeat with second PSF
            Image_2 = np.zeros((N_img, h.Nx, h.Nx))
            I_on_2 = generate_images_photons(h2, [x, y, z, s, 0], N_img = N_on)  
            Image_2[idx,:,:] = I_on_2
            I_sum_2 += Image_2         
        
        
    I_sum += bg # add bg to all images. Molecules are generated with bg = 0
    I_sum += np.random.normal(0, 1, I_sum.shape) #add noise
    I_sum = I_sum * cam.QE / cam.amp  # from photons to counts
    I_sum += cam.baseline
    I_sum[I_sum < cam.baseline] = cam.baseline

    if biplane : 
        I_sum_2 += bg
        I_sum_2 += np.random.normal(0, 1, I_sum_2.shape)
        I_sum_2 = I_sum_2 * cam.QE / cam.amp
        I_sum_2 += cam.baseline
        I_sum_2[I_sum_2 < cam.baseline] = cam.baseline
        I_sum = np.concatenate((I_sum, I_sum_2), axis = 2)
    np.random.shuffle(I_sum) # shuffle the molecule images

    return np.uint16(np.round(I_sum))




def generate_npc(h, cam,
                s = 5000,
                bg = 10,
                N_img = 1000,
                N_mol = 8,
                z0 = 0, # nm
                R = 60,
                dist = 50,
                on_fraction=0.1):
    """
    Generate synthetic SMLM image stack of a signle Nuclear Pore Complex (NPC).
    NPCs are structures made of 2 layers distanced by ~50 nm. 
    Every layer has a diameter of ~120 nm with 8 molecules in a circle.
    """

    I_sum = np.zeros((N_img, h.Nx, h.Nx)) # images are added to this stack
    N_on = int(round(N_img*on_fraction)) #number of images with the molecule on
    I_on = np.zeros((N_on, h.Nx, h.Nx)) # images with molecules

    N_mol = N_mol # molecules per layer (8 for NPC)
    R = R # radius of the circle in nm (60 for NPC)
    dist = dist # distance between the two layers in nm (50 for NPC)
    #phi = np.random.uniform(0, 2*np.pi) # random phase of the circle
    phi = 0

    for layer in range(2) : # layers
        for m in range(1, N_mol+1): # molecules

            Image = np.zeros((N_img, h.Nx, h.Nx))
            idx = np.random.choice(N_img, N_on, replace=False) #randomly choose N_on indices

            # x, y, z positions
            x = R*np.cos(phi + m*2*np.pi/N_mol)*1e-9 #molecule positions are distributed on a circle
            y = R*np.sin(phi + m*2*np.pi/N_mol)*1e-9
            z = (z0 + layer * dist)*1e-9

            I_on = generate_images_photons(h, [x, y, z, s, 0], N_img = N_on) # generate the image without bg
            Image[idx,:,:] = I_on
            I_sum += Image #add to the other molecule images

    # convert photons to counts
    I_sum += bg # add bg to all images here. In the previous step images are generated with bg = 0
    I_sum += np.random.normal(0, 1, I_sum.shape) #add gaussian noise: readout noise
    I_sum = I_sum * cam.QE / cam.amp  # from photons to counts
    I_sum += cam.baseline
    I_sum[I_sum < cam.baseline] = cam.baseline # clip to baseline
    np.random.shuffle(I_sum) # shuffle the molecule images

    return np.uint16(np.round(I_sum))


# GENERATE SINGLE IMAGES

def generate_images_photons(h, param, N_img=1):
    """generating a realistic camera image from a PSF model"""

    dx, dy, dz, s, bg = param
    dx /= h.ux/h.os
    dy /= h.ux/h.os
    dz = (dz - h.z_min) / h.uz 
    itp = jax.image.scale_and_translate(h.data, 
                                        shape = (h.Nx_os, h.Nx_os, 1), #shape of images to be interpolated
                                        spatial_dims = (0, 1, 2), 
                                        scale = jnp.array([1, 1, 1]), 
                                        translation = jnp.array([dx, dy, -dz]),#.clip(v, 0, n-1),
                                        method = 'lanczos5')[...,0]
    
    I_binned = binning(itp, h.os) #binning the interpolated image to the original size
    Image = bg + s * I_binned # the bg input is 0. So no bg added here.
    I_stack = np.repeat(Image[np.newaxis, :, :], N_img, axis=0)
    I_stack = np.uint16(np.random.poisson(I_stack)) # poissonian distribution of photons
    return I_stack




#------------------------------------------------
#             ZERNIKE POLYNOMIALS               #
#------------------------------------------------

# - zernike_noll
# - zernike_nm
# - zernikeRadialFunc
# - zernIndex


def zernike_noll(j, N, rot=0):
    """
      Creates the Zernike polynomial with mode index j,
      where j = 1 corresponds to piston.
      Args:
         j (int): The noll j number of the zernike mode
         N (int): The diameter of the zernike more in pixels
         rot (float): Rotates the Zernike mode by rot radians around  its centre. Defaults to zero.
      Returns:
         ndarray: The Zernike mode
    """
    n, m = zernIndex(j)
    return zernike_nm(n, m, N, rot)


def zernike_nm(n, m, N, rot=0):
    """
      Creates the Zernike polynomial with radial index, n, and  azimuthal index, m.
      Args:
         n (int): The radial order of the zernike polynomial
         m (int): The azimuthal order of the zernike polynomial
         N (int): The diameter of the zernike polynomial in pixels
         rot (float): Rotates the zernike by rot radians around its centre. Defaults to 0.
      Returns:
         ndarray: The Zernike polynomial
    """
    coords = np.fft.fftfreq(N, 1/N) /(N/2) # here is the new definition of the coordinate grid
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    if m==0:
        Z = np.sqrt(n+1)*zernikeRadialFunc(n, 0, R)
    else:
        if m > 0: # j is even
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.cos((m*theta)+rot)
        else:   #i is odd
            m = abs(m)
            Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n, m, R) * np.sin((m*theta)+rot)

    # clip
    Z = Z*np.less_equal(R, 1.0)

    # pupil
    mask = (X**2 + Y**2 <= N) # new!!

    return np.fft.fftshift(Z*mask) # new!!


def zernikeRadialFunc(n, m, r):
     """
     Fucntion to calculate the Zernike radial function
     Parameters:
         n (int): Zernike radial order
         m (int): Zernike azimuthal order
         r (ndarray): 2-d array of radii from the centre the array
     Returns:
         ndarray: The Zernike radial function
     """

     R = np.zeros(r.shape)
     # Can cast the below to "int", n,m are always *both* either even or odd
     for i in range(0, int((n - m) / 2) + 1):

         R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                          np.math.factorial(n - i)) /
                          (np.math.factorial(i) *
                           np.math.factorial(int(0.5 * (n + m) - i)) *
                           np.math.factorial(int(0.5 * (n - m) - i))),
                          dtype='float')
     return R


def zernIndex(j):
     """
     Find the [n,m] list giving the radial order n and azimuthal order
     of the Zernike polynomial of Noll index j.
     Parameters:
         j (int): The Noll index for Zernike polynomials
     Returns:
         list: n, m values
     """
     n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
     p = (j-(n*(n+1))/2.)
     k = n%2
     m = int((p+k)/2.)*2 - k
     if m!=0:
         if j%2==0:
             s=1
         else:
             s=-1
         m *= s
     return [n, m]





#%%
# Parula (matlab) colormap      
_parula_rgb_data = [
[0.2422, 0.1504, 0.6603],[0.2444, 0.1534, 0.6728],[0.2464, 0.1569, 0.6847],[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],[0.2522, 0.1689, 0.7179],[0.2540, 0.1732, 0.7286],[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],[0.2594, 0.1854, 0.7610],[0.2611, 0.1893, 0.7719],[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],[0.2661, 0.2011, 0.8043],[0.2676, 0.2052, 0.8148],[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],[0.2717, 0.2184, 0.8439],[0.2729, 0.2231, 0.8528],[0.2740, 0.2280, 0.8612],
[0.2749, 0.2330, 0.8692],[0.2758, 0.2382, 0.8767],[0.2766, 0.2435, 0.8840],[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],[0.2788, 0.2598, 0.9035],[0.2794, 0.2653, 0.9094],[0.2798, 0.2708, 0.9150],
[0.2802, 0.2764, 0.9204],[0.2806, 0.2819, 0.9255],[0.2809, 0.2875, 0.9305],[0.2811, 0.2930, 0.9352],
[0.2813, 0.2985, 0.9397],[0.2814, 0.3040, 0.9441],[0.2814, 0.3095, 0.9483],[0.2813, 0.3150, 0.9524],
[0.2811, 0.3204, 0.9563],[0.2809, 0.3259, 0.9600],[0.2807, 0.3313, 0.9636],[0.2803, 0.3367, 0.9670],
[0.2798, 0.3421, 0.9702],[0.2791, 0.3475, 0.9733],[0.2784, 0.3529, 0.9763],[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],[0.2754, 0.3693, 0.9840],[0.2741, 0.3748, 0.9862],[0.2726, 0.3804, 0.9881],
[0.2710, 0.3860, 0.9898],[0.2691, 0.3916, 0.9912],[0.2670, 0.3973, 0.9924],[0.2647, 0.4030, 0.9935],
[0.2621, 0.4088, 0.9946],[0.2591, 0.4145, 0.9955],[0.2556, 0.4203, 0.9965],[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],[0.2424, 0.4378, 0.9991],[0.2369, 0.4437, 0.9996],[0.2311, 0.4497, 0.9995],
[0.2250, 0.4559, 0.9985],[0.2189, 0.4620, 0.9968],[0.2128, 0.4682, 0.9948],[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],[0.1950, 0.4861, 0.9887],[0.1903, 0.4919, 0.9867],[0.1869, 0.4975, 0.9844],
[0.1847, 0.5030, 0.9819],[0.1831, 0.5084, 0.9793],[0.1818, 0.5138, 0.9766],[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],[0.1785, 0.5296, 0.9677],[0.1778, 0.5349, 0.9641],[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.9560],[0.1764, 0.5504, 0.9516],[0.1755, 0.5554, 0.9473],[0.1740, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],[0.1686, 0.5705, 0.9357],[0.1649, 0.5755, 0.9323],[0.1610, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],[0.1540, 0.5902, 0.9218],[0.1513, 0.5950, 0.9182],[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],[0.1461, 0.6089, 0.9080],[0.1446, 0.6135, 0.9050],[0.1429, 0.6180, 0.9022],
[0.1408, 0.6226, 0.8998],[0.1383, 0.6272, 0.8975],[0.1354, 0.6317, 0.8953],[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.8910],[0.1253, 0.6453, 0.8887],[0.1219, 0.6497, 0.8862],[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],[0.1119, 0.6627, 0.8770],[0.1085, 0.6669, 0.8734],[0.1048, 0.6710, 0.8695],
[0.1009, 0.6750, 0.8653],[0.0964, 0.6789, 0.8609],[0.0914, 0.6828, 0.8562],[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],[0.0713, 0.6938, 0.8409],[0.0628, 0.6972, 0.8355],[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],[0.0328, 0.7071, 0.8183],[0.0234, 0.7103, 0.8124],[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],[0.0046, 0.7192, 0.7941],[0.0019, 0.7220, 0.7878],[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],[0.0046, 0.7301, 0.7688],[0.0094, 0.7327, 0.7623],[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],[0.0369, 0.7400, 0.7426],[0.0504, 0.7423, 0.7359],[0.0638, 0.7446, 0.7292],
[0.0770, 0.7468, 0.7224],[0.0899, 0.7489, 0.7156],[0.1023, 0.7510, 0.7088],[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.6950],[0.1354, 0.7572, 0.6881],[0.1448, 0.7593, 0.6812],[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],[0.1678, 0.7656, 0.6599],[0.1741, 0.7678, 0.6527],[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],[0.1905, 0.7743, 0.6303],[0.1954, 0.7765, 0.6225],[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],[0.2118, 0.7828, 0.5983],[0.2178, 0.7849, 0.5899],[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],[0.2401, 0.7905, 0.5636],[0.2491, 0.7922, 0.5546],[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.5360],[0.2809, 0.7964, 0.5266],[0.2929, 0.7975, 0.5170],[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],[0.3301, 0.8002, 0.4876],[0.3424, 0.8009, 0.4774],[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],[0.3795, 0.8026, 0.4454],[0.3921, 0.8029, 0.4344],[0.4050, 0.8031, 0.4233],
[0.4184, 0.8030, 0.4122],[0.4322, 0.8028, 0.4013],[0.4463, 0.8024, 0.3904],[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],[0.4899, 0.8002, 0.3586],[0.5044, 0.7993, 0.3480],[0.5187, 0.7982, 0.3374],
[0.5329, 0.7970, 0.3267],[0.5470, 0.7957, 0.3159],[0.5609, 0.7943, 0.3050],[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],[0.6024, 0.7896, 0.2726],[0.6161, 0.7878, 0.2622],[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],[0.6567, 0.7818, 0.2329],[0.6701, 0.7796, 0.2239],[0.6833, 0.7773, 0.2155],
[0.6963, 0.7750, 0.2075],[0.7091, 0.7727, 0.1998],[0.7218, 0.7703, 0.1924],[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],[0.7590, 0.7629, 0.1717],[0.7710, 0.7604, 0.1658],[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.1570],[0.8060, 0.7529, 0.1546],[0.8172, 0.7505, 0.1535],[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],[0.8495, 0.7435, 0.1564],[0.8600, 0.7413, 0.1587],[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.1650],[0.8903, 0.7353, 0.1695],[0.9000, 0.7336, 0.1749],[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.1890],[0.9272, 0.7298, 0.1973],[0.9357, 0.7290, 0.2061],[0.9440, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],[0.9606, 0.7285, 0.2312],[0.9689, 0.7292, 0.2373],[0.9770, 0.7304, 0.2418],
[0.9842, 0.7330, 0.2446],[0.9900, 0.7365, 0.2429],[0.9946, 0.7407, 0.2394],[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],[0.9972, 0.7569, 0.2267],[0.9971, 0.7626, 0.2224],[0.9969, 0.7683, 0.2181],
[0.9966, 0.7740, 0.2138],[0.9962, 0.7798, 0.2095],[0.9957, 0.7856, 0.2053],[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],[0.9923, 0.8034, 0.1939],[0.9906, 0.8095, 0.1906],[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],[0.9835, 0.8280, 0.1817],[0.9807, 0.8342, 0.1787],[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],[0.9720, 0.8529, 0.1695],[0.9694, 0.8591, 0.1665],[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],[0.9634, 0.8778, 0.1582],[0.9619, 0.8840, 0.1557],[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],[0.9596, 0.9023, 0.1480],[0.9595, 0.9084, 0.1450],[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],[0.9608, 0.9262, 0.1344],[0.9618, 0.9320, 0.1304],[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],[0.9657, 0.9494, 0.1168],[0.9674, 0.9552, 0.1116],[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],[0.9730, 0.9724, 0.0938],[0.9749, 0.9782, 0.0872],[0.9769, 0.9839, 0.0805]]
from matplotlib.colors import ListedColormap
cmaps = {}
cmaps['parula'] = ListedColormap(_parula_rgb_data, name='parula')
parula = cmaps['parula']

# %%
