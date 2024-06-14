from quTARANG.config.config import ncp
from quTARANG import para
from quTARANG.src import evolution
from quTARANG.src.univ import fns, grid, data_io
from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.lib import vector_field as U
from quTARANG.util import fns_util as util
from pathlib import Path 
import shutil
import os 
import sys

class GPE():
    """Main class for the simulation.
    """
    def __init__(self, wfcfn = None, potfn = None):
        self.wfc = []
        self.wfck = []
        self.pot = []
        #list of all type of energies
        self.te = []
        self.ke = []
        self.kec = []
        self.kei = []
        self.pe = []
        self.qe = []
        self.ie = []
        self.chmpot = []
        self.estime = []
        
        self.xrms = []
        self.yrms = []
        self.zrms = []
        self.rrms = []
        self.srmstime = []
        
        self.normfact = para.Npar
        self.omega = para.omega
        self.wfcfn = wfcfn
        self.potfn = potfn 
        
        
        self.U = U.Vector_field()
        data_io.show_params()
        self.set_arrays()
        
        self.set_initfile()
        
        # For timedependent potential and omega
        if para.resume == True: 
            data_io.set_resume_initcond(self)
            
        elif para.resume == False:
            data_io.set_initcond(self)
        else:
            sys.exit("resume in para.py can take the boolean value only!")
        self.renorm()   
        self.wfck[:] = fft.forward_transform(self.wfc)
        
        evolution.set_scheme(self)      
    
    def set_arrays(self):
        """Setup numpy/cupy arrays for wfc, wfck and pot for the simulation.
        """
        if para.dimension == 1:
            self.wfc = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.wfck = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.pot = ncp.zeros(para.Nx, dtype = para.real_dtype)
            
        elif para.dimension == 2:
            self.wfc = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.wfck = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.pot = ncp.zeros((para.Nx, para.Ny), dtype = para.real_dtype)
        
        elif para.dimension == 3:
            self.wfc = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)   
            self.wfck = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)      
            self.pot = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.real_dtype)  
    
    def set_initfile(self):
        data_io.gen_path()
        path = Path(para.op_path)/'para.py'
        
        if para.overwrite == False:
            if (path.is_file() and (not para.resume)):
                sys.exit("Data for a run is already in the output path. \nEither move whole data or change the folder!")
        
        else:
            shutil.copy(Path(os.path.realpath(__file__)).parents[2]/'para.py', Path(para.op_path)/'para.py')
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'main.py', Path(para.op_path)/'main.py')
            
        if not (path.is_file()):
            shutil.copy(Path(os.path.realpath(__file__)).parents[2]/'para.py', Path(para.op_path)/'para.py')
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'main.py', Path(para.op_path)/'main.py')
    
    # computes the norm
    def renorm(self):   #C
        self.wfc = self.normfact * self.wfc/self.norm()**0.5

    def norm(self): 
        return fns.integralr(ncp.abs(self.wfc)**2)
    
    def norm_wfc(self, psi):
        return fns.integralr(ncp.abs(psi)**2)
    
    def evolve(self):
        evolution.time_advance(self)

    def compute_chmpot(self):
        self.U.temp[:] = fft.forward_transform(self.wfc)
        deriv = grid.volume * ncp.sum(grid.ksqr * ncp.abs(self.U.temp)**2)
        return fns.integralr(((self.pot + para.g * ncp.abs(self.wfc)**2) * ncp.abs(self.wfc)**2)) + deriv/2 - para.omega * self.compute_Lz()

    def compute_xrms(self):  
        return (fns.integralr(ncp.abs(self.wfc)**2 * grid.x_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.x_mesh))**2)**.5

    def compute_yrms(self):  
        return (fns.integralr(ncp.abs(self.wfc)**2 * grid.y_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.y_mesh))**2)**.5   
    
    def compute_zrms(self):  
        return (fns.integralr(ncp.abs(self.wfc)**2 * grid.z_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.z_mesh))**2)**.5   

    def compute_rrms(self):
        if para.dimension == 2:   
            return (fns.integralr(ncp.abs(self.wfc) ** 2 * (grid.x_mesh**2 + grid.y_mesh**2)) - (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2)**0.5)))**.5 
        elif para.dimension == 3:
            return (fns.integralr(ncp.abs(self.wfc) ** 2 * (grid.x_mesh**2 + grid.y_mesh**2 + grid.z_mesh**2)) - (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2 + grid.z_mesh**2)**0.5)))**.5

    def compute_Lz(self):
        self.U.temp[:] = 1j * (grid.y_mesh * fns.xderivative(self.wfc) - grid.x_mesh * fns.yderivative(self.wfc))
        return (fns.integralr(self.U.temp * self.wfc.conj())).real

    def compute_te(self):
        self.U.temp[:] = fft.forward_transform(self.wfc) #for sstep_strang
        # deriv = (fns.integralr(self.wfc.conj() * fft.inverse_transform(self.U.temp * grid.ksqr))).real
        deriv = grid.volume * ncp.sum(grid.ksqr * ncp.abs(self.U.temp)**2)
        return fns.integralr(((self.pot + 0.5 * para.g * ncp.abs(self.wfc)**2) * ncp.abs(self.wfc)**2)) + deriv/2 - para.omega * self.compute_Lz()

    '''
    def compute_quantum_energy(self):
        fns.gradient(self.params, ncp.abs(self.wfc), self)
        if grid.dimension == 2:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2)

        elif grid.dimension == 3:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integralr(self.params, self.U.temp.real, self.grid)


    def compute_internal_energy(self):
        return 0.5 * self.params.g * fns.integralr(self.params, ncp.abs(self.wfc)**4, self.grid)


    def compute_potential_energy(self):
        return fns.integralr(self.params, self.pot * ncp.abs(self.wfc)**2, self.grid)

    def compute_velocity(self):
        fns.gradient(self.params, self.wfc.conj(), self)
        if grid.dimension == 2:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2

        elif grid.dimension == 3:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2
            self.U.Vz[:] = -(self.wfc * self.U.Vz).imag/ncp.abs(self.wfc)**2
        return


    def compute_kinetic_energy(self):  # doubt while integrating in fourier space
        self.compute_velocity()
        self.U.temp[:] = 0.5 * ncp.abs(self.wfc)**2 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integralr(self.params, self.U.temp.real, self.grid)

    def omegak(self):
        self.compute_velocity()
        self.U.temp[:] = ncp.abs(self.wfc)
        self.U.omegai_kx[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vx)
        self.U.omegai_ky[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vy)

        if grid.dimension == 2:
            self.grid.ksqr[0, 0] = 1
            self.U.temp[:] = (self.grid.kxx * self.U.omegai_kx + self.grid.kyy * self.U.omegai_ky)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp
            self.grid.ksqr[0, 0] = 0

            #incompressible part calculation
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy

        elif grid.dimension == 3:
            self.U.omegai_kz[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vz)
            self.grid.ksqr[0, 0, 0] = 1
            self.U.temp[:] = (self.grid.kxx * self.U.omegai_kx + self.grid.kyy * self.U.omegai_ky + self.grid.kzz * self.U.omegai_kz)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp
            self.U.Vz[:] = self.grid.kzz * self.U.temp
            self.grid.ksqr[0, 0, 0] = 0
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
            self.U.omegai_kz[:] = self.U.omegai_kz - self.U.Vz
        return


    def KE_decomp(self):
        """
        This function calculates the kinetic energy decomposition

        Returns
        -------
        arrays
            arrays containing the kinetic energy decomposition
        """
        self.omegak()
        if grid.dimension == 2:
            KE_comp = 0.5 * fns.integralk(self.params, ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2, self.params)
            KE_incomp = 0.5 * fns.integralk(self.params, ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2, self.params)
        else:
            KE_comp = 0.5 * fns.integralk(self.params, ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2 + ncp.abs(self.U.Vz)**2, self.params)
            KE_incomp = 0.5 * fns.integralk(self.params, ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2 + ncp.abs(self.U.omegai_kz)**2, self.params)
        return KE_comp, KE_incomp


    # For calculation of particle number flux
    # def compute_tk_particle_no(self):
    #     self.U.temp[:] = my_fft.forward_transform(self.params, self.wfc)
    #     self.U.temp1[:] = my_fft.forward_transform(self.params, self.params.g * self.wfc * ncp.abs(self.wfc)**2 + self.wfc * self.pot)
    #     temp = (self.U.temp1[:] * ncp.conjugate(self.U.temp)).imag
    #     return self.binning(temp)

    # def comp_par_no_spectrum(self):
    #     self.U.temp[:] = ncp.abs(my_fft.forward_transform(self.params, self.wfc))**2
    #     return self.bining(self.U.temp)

    def comp_KEcomp_spectrum(self):
        self.omegak()
        if grid.dimension == 2:
            KE_incompk = 0.5 * (ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2)
            KE_compk = 0.5 * (ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2)
        elif grid.dimension == 3:
            KE_incompk = 0.5 * (ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2 + ncp.abs(self.U.omegai_kz)**2)
            KE_compk = 0.5 * (ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2 + ncp.abs(self.U.Vz)**2)

        KEcomp_spectrum = self.binning(KE_compk)
        KEincomp_spectrum = self.binning(KE_incompk)
        # print (KE_compk[2], KE_incompk[2])
        return KEcomp_spectrum, KEincomp_spectrum



    def binning(self, quantity):
        quantity_s = ncp.zeros(para.Nx//2)
        for i in range(para.Nx//2):
            z = ncp.where((self.grid.ksqr**.5 >= self.grid.kxx[i]) & (self.grid.ksqr**.5 < self.grid.kxx[i+1]))
            quantity_s[i] = ncp.sum(quantity[z])
        return quantity_s
'''