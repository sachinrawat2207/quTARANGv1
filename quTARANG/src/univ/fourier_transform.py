# from quTARANG.lib_multigpu_fft import my_fft
from quTARANG import para
from quTARANG.config.config import fft

def forward_transform1D(psi, axis = -1):
    return fft.fft(psi, axis = axis)/(para.Nx)

def inverse_transform1D(psik, axis = -1):
    return fft.ifft(psik, axis = axis) * (para.Nx)

def forward_transform2D(psi):
    return fft.fft2(psi)/(para.Nx * para.Ny)

def inverse_transform2D(psik):
    return fft.ifft2(psik) * (para.Nx * para.Ny)

def forward_transform3D(psi):
    return fft.fftn(psi)/(para.Nx * para.Ny * para.Nz)

def inverse_transform3D(psik):
    return fft.ifftn(psik) * (para.Nx * para.Ny * para.Nz)

if para.dimension == 1:
    forward_transform = forward_transform1D
    inverse_transform = inverse_transform1D

elif para.dimension == 2:
    forward_transform = forward_transform2D
    inverse_transform = inverse_transform2D

elif para.dimension == 3:
    forward_transform = forward_transform3D
    inverse_transform = inverse_transform3D