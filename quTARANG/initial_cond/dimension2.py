from quTARANG.config.config import ncp
from quTARANG import para 
from quTARANG.src.univ import grid
ncp.random.seed(0)

def test():
    gammay = 2
    eps1 = 2
    wfc = ((gammay)**0.25/(ncp.pi * eps1)**(0.5)) * ncp.exp(-(gammay * grid.y_mesh**2 + grid.x_mesh**2)/(2*eps1)) + 0j
    pot = (grid.x_mesh**2 + gammay**2 * grid.y_mesh**2)/2
    return wfc, pot

def srp():  
    # Generate the smoothed random phase 
    theta_0 = 1
    
    dk = 2*ncp.pi/para.Lx
    kx = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    ky = ncp.arange(para.Ny//2+1)*dk

    kx_mesh, ky_mesh = ncp.meshgrid(kx, ky, indexing='ij')
    kmod = ncp.sqrt(kx_mesh**2 + ky_mesh**2)
    
    # Chooses theta_kx between -pi to pi
    theta_kx = 0.9999*ncp.pi * (ncp.random.random((para.Nx, para.Ny//2+1))*2 - 1)
    theta = theta_0 * ncp.exp(1j * theta_kx)
    theta[1:para.Nx//2,0] = ncp.conj(theta[para.Nx-1:para.Nx//2:-1, 0])

    z = ncp.where((kmod > 3*dk) | (kmod < dk))
    theta[z] = 0

    phase = ncp.fft.irfft2(theta)*para.Nx*para.Ny

    wfc = ncp.exp(1j * phase)
    pot = 0

    return wfc, pot

# Initial conditon for rotating frame of reference
def rot_centralgaussian():
    gamma_x = 1
    gamma_y = 1

    pot = 0.5*(gamma_x**2*grid.x_mesh**2 + gamma_y**2*grid.y_mesh**2) 

    phiho = 1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)
    phihonu =  1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)*(grid.x_mesh - 1j*grid.y_mesh)
    wfc = (1 - para.omega)*phiho + para.omega*phihonu
    return wfc, pot

def rot_centralgaussianadhi():
    gamma_x = 1
    gamma_y = 1
    beta = 20
    V  = ncp.zeros_like(grid.x_mesh)
    theta = ncp.arctan2(grid.y_mesh, grid.x_mesh)
    # theta = ncp.arctan(grid.y_mesh/grid.x_mesh)
    A = 1.25
    R = .5
    v0 = 20
    alpha = 10
    delta = 0.4
    r = ncp.sqrt(grid.x_mesh**2 + grid.y_mesh**2)
    rval = R + A*(ncp.sin(alpha* theta)+ncp.sin(beta*theta+delta))
    V[ncp.where(r<rval)] = v0
    
    pot = 0.5*(gamma_x**2*grid.x_mesh**2 + gamma_y**2*grid.y_mesh**2) + V
    phiho = 1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)
    phihonu =  1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)*(grid.x_mesh - 1j*grid.y_mesh)
    wfc = (1 - para.omega)*phiho + para.omega*phihonu
    return wfc, pot