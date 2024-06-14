from quTARANG.config.config import  ncp
from quTARANG import para

class Vector_field:
    """ It contains the variables which helps to calculate the different physical quantities of GPE class 
    """
    Vx = []
    Vy = []
    Vz = []
    # Used to calculate omegai
    omegai_kx = []
    omegai_ky = []
    omegai_ky = []
    
    # Temporary variable
    temp = []
    
    def __init__(self) -> None:
        self.set_arrays()
        
    def set_arrays(self) -> None:
        if para.dimension == 1:
            self.Vx = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp = ncp.zeros(para.Nx, dtype = para.complex_dtype) # used
            self.temp1k = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp2k = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp3k = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp4k = ncp.zeros(para.Nx, dtype = para.complex_dtype)

        elif para.dimension == 2:
            self.Vx = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.Vy = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.omegai_ky = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp1k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp2k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp3k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp4k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
        
        elif para.dimension == 3:
            self.Vx = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.Vy = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.Vz = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_ky = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_kz = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp1k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp2k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp3k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp4k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)