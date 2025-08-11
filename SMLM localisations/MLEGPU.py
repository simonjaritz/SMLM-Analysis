import numpy as np
import cupy as cp
from numba import njit
import os

@njit()
def prefilter_forward_rows(data, M=12):
    nx,ny,nz = data.shape
    dtp = data.dtype
    zp = np.sqrt(3) - 2.
    lbda = 6.
    cp = np.zeros((nx, ny, nz),dtype=dtp)
    cp0 = data[0]
    pf = 1. / (1. - (zp ** 2) ** nx)

    # initialize causal

    for k in range(M):
        cp0 += pf * (zp ** (k + 1) + zp ** (2 * nx - k)) * data[k]
    cp[0] = cp0 * lbda

    for k in range(1, nx):
        cp[k] = lbda * data[k] + zp * cp[k - 1]

    cm = np.zeros((nx, ny, nz),dtype=dtp)
    cm[-1] = -zp / (1. - zp) * cp[-1]

    for k in np.arange(nx - 2, -1, -1):
        cm[k] = zp * (cm[k + 1] - cp[k])
    return cm


@njit(cache=True)
def prefilter_forward(data, M=12):

    cm_r = prefilter_forward_rows(data, M=M)
    cm_c = prefilter_forward_rows(cm_r.transpose(1, 0, 2), M=M)
    cm_d = prefilter_forward_rows(cm_c.transpose(2, 1, 0), M=M)

    return cm_d.transpose(1, 2, 0)
# todo: channels, pinned memory, different streams

class MLEFitLMGPU(object):

    def __init__(self,
                 grid,
                 grid_psf,
                 os,
                 batch_size):
        
        assert len(grid) == 2
        assert len(grid_psf) == 3

        self.nx,self.ny = grid
        self.nx_psf,self.ny_psf,self.nz_psf = grid_psf
        self.npar = 5
        self.os = np.int32(os)
        self.batch_size = np.int32(batch_size)
        self.dtp = cp.float32

        assert self.nx * self.os == self.nx_psf, f"os * grid[0] is not equal to grid_psf[0]"
        assert self.ny * self.os == self.ny_psf, f"os * grid[1] is not equal to grid_psf[1]"
        assert 0 < self.batch_size < 2**16, f"choose batch_size smaller than 2^16 = 65536"

        self.__init_kernels()
        self.__init_texture()
        self.__arrays()

    def __init_kernels(self):
        self.pth = os.path.dirname(__file__)
        self.prg = cp.RawModule(code = open(os.path.join(self.pth,'lm_fit_kernels.cu'),'r').read())
        self.prg.compile()

        #self.tex_ref_PSF = self.prg.get_texref('im_PSF') deprecated
        self.comp_J_CP = self.prg.get_function('comp_J')

    def __init_texture(self):

        self.ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
        self.arr_psf = cp.cuda.texture.CUDAarray(self.ch, self.nz_psf, self.ny_psf, self.nx_psf)
        self.res_psf = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=self.arr_psf)
        self.address_mode = cp.cuda.runtime.cudaAddressModeClamp
        self.tex_psf = cp.cuda.texture.TextureDescriptor((self.address_mode,
                                                          self.address_mode,
                                                          self.address_mode),
                                                          cp.cuda.runtime.cudaFilterModeLinear,
                                                          cp.cuda.runtime.cudaReadModeElementType)
        self.tex_obj_psf = cp.cuda.texture.TextureObject(self.res_psf, self.tex_psf)
        #self.tex_obj_psf = cp.cuda.texture.TextureReference(self.tex_ref_PSF, self.res_psf, self.tex_psf)

    def __arrays(self):

        self.J_cp = cp.zeros((self.batch_size,self.nx*self.ny,self.npar),dtype=self.dtp)
        self.JTJ_cp = cp.zeros((self.batch_size,self.npar,self.npar),dtype=self.dtp)
        self.residual_cp = cp.zeros((self.batch_size,self.nx*self.ny,1),dtype=self.dtp)
        self.JTr_cp = cp.zeros((self.batch_size,self.npar,1),dtype=self.dtp)
        self.err_poi_cp = cp.zeros((self.batch_size,self.nx*self.ny),dtype=self.dtp)
        self.fac_jac_cp = cp.zeros((self.batch_size,self.nx*self.ny,1),dtype=self.dtp)
        self.M_cp = cp.zeros((self.batch_size,self.nx,self.ny),dtype=self.dtp)
        self.M_modif_cp = cp.zeros((self.batch_size, self.nx, self.ny), dtype=self.dtp)
        self.chi2_cp = cp.zeros((self.batch_size),dtype=self.dtp)
        self.chi2_old_cp = cp.zeros((self.batch_size),dtype=self.dtp)
        self.lbda_cp = cp.zeros((self.batch_size),dtype=self.dtp)

        self.x0_cp = cp.zeros(self.batch_size, dtype=self.dtp)
        self.y0_cp = cp.zeros(self.batch_size, dtype=self.dtp)
        self.z0_cp = cp.zeros(self.batch_size, dtype=self.dtp)
        self.BG_cp = cp.zeros(self.batch_size, dtype=self.dtp)
        self.NP_cp = cp.zeros(self.batch_size, dtype=self.dtp)

    def set_psf(self,psf_array):

        assert psf_array.ndim == 3
        assert psf_array.shape == (self.nx_psf,self.ny_psf,self.nz_psf)
        self.arr_psf.copy_from(prefilter_forward(psf_array).astype(cp.float32))

    def fit(self,
            x0_start,
            y0_start,
            z0_start,
            BG_start,
            NP_start,
            M,
            bounds,
            maxiter=10):

        self.nw = x0_start.shape[0]
        if self.nw % self.batch_size == 0:
            self.last_step_partial = False
            self.num_runs = self.nw//self.batch_size
        else:
            self.last_step_partial = True
            self.num_runs = self.nw // self.batch_size + 1

        self.x0min, self.x0max = bounds[0]
        self.y0min, self.y0max = bounds[1]
        self.z0min, self.z0max = bounds[2]
        self.BGmin, self.BGmax = bounds[3]
        self.NPmin, self.NPmax = bounds[4]

        self.x0_result = np.zeros_like(x0_start)
        self.y0_result = np.zeros_like(y0_start)
        self.z0_result = np.zeros_like(z0_start)
        self.BG_result = np.zeros_like(BG_start)
        self.NP_result = np.zeros_like(NP_start)
        self.chi2_result = np.zeros_like(x0_start)

        self.evt = cp.cuda.Event()

        for i in range(self.num_runs):
            self.start = i * self.batch_size
            if self.last_step_partial & (i == self.num_runs-1):
                self.stop = self.nw

            else:
                self.stop = (i+1) * self.batch_size

            self.set_batch_to(x0_start,
                              y0_start,
                              z0_start,
                              BG_start,
                              NP_start,
                              M,
                              start=self.start,
                              stop=self.stop)
            self.fit_batch(self.x0_cp,
                           self.y0_cp,
                           self.z0_cp,
                           self.BG_cp,
                           self.NP_cp,
                           self.M_cp,
                           maxiter=maxiter)
            self.evt.synchronize()
            self.set_batch_from(self.x0_result,
                                self.y0_result,
                                self.z0_result,
                                self.BG_result,
                                self.NP_result,
                                start=self.start,
                                stop=self.stop)
        self.results = {'x0':self.x0_result,
                        'y0':self.y0_result,
                        'z0':self.z0_result,
                        'BG':self.BG_result,
                        'NP':self.NP_result,
                        'chi2':self.chi2_result}
        return self.results
    
    def set_batch_from(self,
                       x0_result,
                       y0_result,
                       z0_result,
                       BG_result,
                       NP_result,
                       start,
                       stop):

        self.sl_b = slice(start, stop)
        self.x0_result[self.sl_b] = self.x0_cp.get()[:(stop-start)]
        self.y0_result[self.sl_b] = self.y0_cp.get()[:(stop-start)]
        self.z0_result[self.sl_b] = self.z0_cp.get()[:(stop-start)]
        self.BG_result[self.sl_b] = self.BG_cp.get()[:(stop-start)]
        self.NP_result[self.sl_b] = self.NP_cp.get()[:(stop-start)]
        self.chi2_result[self.sl_b] = self.chi2_cp.get()[:(stop-start)]

    def set_batch_to(self,
                     x0_start,
                     y0_start,
                     z0_start,
                     BG_start,
                     NP_start,
                     M,
                     start,
                     stop):

        self.sl_b = slice(start, stop)
        self.x0_cp[:(stop-start)].set(x0_start[self.sl_b][:(stop-start)])
        self.y0_cp[:(stop-start)].set(y0_start[self.sl_b][:(stop-start)])
        self.z0_cp[:(stop-start)].set(z0_start[self.sl_b][:(stop-start)])
        self.BG_cp[:(stop-start)].set(BG_start[self.sl_b][:(stop-start)])
        self.NP_cp[:(stop-start)].set(NP_start[self.sl_b][:(stop-start)])
        self.M_cp[:(stop-start)].set(M[self.sl_b][:(stop-start)])

    def fit_batch(self,
                   x0_cp,
                   y0_cp,
                   z0_cp,
                   BG_cp,
                   NP_cp,
                   M_cp,
                   maxiter = 10):
        self.iteration = 0
        self.Lup = 9
        self.Ldown = 11
        self.lbda_max = 1e7
        self.lbda_min = 1e-7
        self.lbda_cp.fill(1)
        
        self.grs = (1,self.nx,self.batch_size)
        self.bls = (self.ny,1,1)
        self.chi2_cp.fill(0)
        self.chi2_old_cp.fill(0)

        while self.iteration < maxiter:
            
            self.comp_J_CP(self.grs,self.bls,
                        (self.tex_obj_psf,
                         self.J_cp,
                         self.residual_cp,
                         self.err_poi_cp,
                         M_cp,
                         x0_cp,
                         y0_cp,
                         z0_cp,
                         BG_cp,
                         NP_cp,
                         self.M_modif_cp,
                         self.fac_jac_cp,
                         np.int32(self.os),
                         np.float32(self.nz_psf),
                         np.int32(self.npar))
                           )
            # copy old error to compare
            cp.copyto(self.chi2_old_cp,self.chi2_cp)
            # compute chi2 for each image
            cp.sum(self.err_poi_cp,axis=1,out=self.chi2_cp)
            # compute LM parameter for every image
            # increase if error increased, decrease otherwise
            self.lbda_cp[:] = cp.where(self.chi2_cp > self.chi2_old_cp,
                                       cp.minimum(self.lbda_cp*self.Lup,self.lbda_max),
                                       cp.maximum(self.lbda_cp/self.Ldown,self.lbda_min))
            # compute Jacobian.Transpose * residual
            self.JTr_cp[:] = cp.einsum('nkj,nkl->njl',self.J_cp,self.residual_cp)
            # multiply for Poissonian Estimator
            cp.multiply(self.J_cp,self.fac_jac_cp,out=self.J_cp)
            # compute Jacobian.Transpose * Jacobian
            self.JTJ_cp[:] = cp.einsum('nkj,nkl->njl',self.J_cp,self.J_cp)
            # add lbda * I
            self.JTJ_cp[:,np.arange(self.npar),np.arange(self.npar)] += self.lbda_cp[:,np.newaxis] * self.JTJ_cp[:,np.arange(self.npar),np.arange(self.npar)]
            # solve system to get updates
            self.dpar_cp = cp.linalg.solve(self.JTJ_cp,self.JTr_cp)
            # todo: trf instead of LM
            # enforce box-constraints
            cp.clip(self.x0_cp - self.dpar_cp[:, 0, 0], self.x0min, self.x0max, out=self.x0_cp)
            cp.clip(self.y0_cp - self.dpar_cp[:, 1, 0], self.y0min, self.y0max, out=self.y0_cp)
            cp.clip(self.z0_cp - self.dpar_cp[:, 2, 0], self.z0min, self.z0max, out=self.z0_cp)
            cp.clip(self.BG_cp - self.dpar_cp[:, 3, 0], self.BGmin, self.BGmax, out=self.BG_cp)
            cp.clip(self.NP_cp - self.dpar_cp[:, 4, 0], self.NPmin, self.NPmax, out=self.NP_cp)
            #self.err = self.err_cp.sum() * 1. / self.batch_size
            self.iteration += 1
            #print(self.iteration,self.err)









