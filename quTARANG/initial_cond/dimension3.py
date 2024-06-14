from quTARANG.config.config import ncp
from quTARANG import para 
from quTARANG.src.univ import grid

ncp.random.seed(1)

def test():
    gammax = 1
    gammay = 2.0
    gammaz = 4.0
    eps1 = 1/4
    wfc = ((gammay * gammaz)**0.25/(ncp.pi * eps1)**(3/4)) * ncp.exp(-( gammaz * grid.z_mesh**2 + gammay * grid.y_mesh**2 + grid.x_mesh**2)/(2*eps1))+0j
    pot = (gammax**2 * grid.x_mesh**2 + gammay**2 * grid.y_mesh**2 + gammaz**2*grid.z_mesh**2)/2 
    return wfc, pot

def srp():  
    # Generate the smoothed random phase 
    theta_0 = 1
    dk = 2*ncp.pi/para.Lx
    
    kx = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    ky = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    kz = ncp.arange(para.Ny//2+1)*dk

    kx_mesh, ky_mesh, kz_mesh = ncp.meshgrid(kx, ky, kz, indexing='ij')
    kmod = ncp.sqrt(kx_mesh**2 + ky_mesh**2+ kz_mesh**2)
    
    # Chooses theta_kx between -pi to pi
    theta_kx = 0.999 * ncp.pi * (ncp.random.random((para.Nx, para.Ny, para.Nz//2+1))*2 - 1)
    theta = theta_0 * ncp.exp(1j * theta_kx)
    
    theta[para.Nx-1:para.Nx//2:-1,para.Ny-1:para.Ny//2:-1,0] = ncp.conj(theta[1:para.Nx//2,1:para.Ny//2,0])
    theta[para.Nx-1:para.Nx//2:-1,para.Ny//2-1:0:-1,0] = ncp.conj(theta[1:para.Nx//2,para.Ny//2+1:para.Ny,0])

    theta[para.Nx-1:para.Nx//2:-1,0,0] = ncp.conj(theta[1:para.Nx//2,0,0])
    theta[0,para.Ny-1:para.Ny//2:-1,0] = ncp.conj(theta[0,1:para.Ny//2,0])


    z = ncp.where((kmod > 3*dk) | (kmod < dk))
    theta[z] = 0
    phase = ncp.fft.irfftn(theta)*para.Nx*para.Ny
    
    # normalize the phase between -4pi to 4pi
    arg_norm = 4*ncp.pi
    phase = 2*arg_norm*(phase - ncp.min(phase))/ncp.ptp(phase)-arg_norm
    
    wfc = ncp.exp(1j * phase)
    pot = 0
    
    return wfc, pot
