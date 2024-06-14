from quTARANG import para
from quTARANG.src.lib import gpe, rgpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp

# def omega_(t):
#     Tr = 120
#     omega_0 = 0.3
#     omegaf = 1.2
#     if t>Tr:
#         return omegaf
#     else:
#         return omega_0+(omegaf-omega_0)*ncp.sin(ncp.pi*(t)/(2*Tr))**2 

# def paddlepot(t):
#     V0 = 50
#     neta = 0.2
#     d = 2
#     x0 = para.Lx/2
#     y0 = para.Ly/2
    
#     omega = 2*ncp.pi/para.Tp
#     omegap = 2*ncp.pi/para.Tosc
#     alpha = 1
#     xtilde = grid.x_mesh - x0 * alpha * ncp.sin(omega*t)
#     ytilde = grid.y_mesh - y0 * alpha * ncp.cos(omega*t)
#     if t <= para.Tf:
#         return V0*ncp.exp(-neta**2*(xtilde * ncp.cos(omegap*t) - ytilde * ncp.sin(omegap*t))**2/d**2 - (ytilde * ncp.cos(omegap*t) + xtilde * ncp.sin(omegap*t))**2/d**2)
    
#     elif t> para.Tf and t<=para.T0:
#         return V0/(para.T0-para.Tf)*(para.T0-t)**ncp.exp(-neta**2*(xtilde * ncp.cos(omegap*t) - ytilde * ncp.sin(omegap*t))**2/d**2 - (ytilde * ncp.cos(omegap*t) + xtilde * ncp.sin(omegap*t))**2/d**2)
    
#     else:
#         return 0

gammay = 1
gammax = 1


def wfc():
    phiho = (gammax * gammay)**(0.25)/(ncp.pi)**0.5 * ncp.exp(-(gammax * grid.x_mesh**2 + gammay * grid.y_mesh**2)/2)
    phinu = (gammax * grid.x_mesh + 1j * gammay * grid.y_mesh)/(ncp.pi)**0.5 * ncp.exp(-(gammax * grid.x_mesh**2 + gammay * grid.y_mesh**2)/2)
    return (1 - para.omega) * phiho + para.omega * phinu

V_trap = 0.5*(gammax**2 * grid.x_mesh**2 + gammay**2 * grid.y_mesh**2)

def pot(t):
    return V_trap

if para.omega == 0 and para.gamma == 0:
    if para.imgtime == True:
        G = gpe.GPE(potfn = pot, wfcfn = wfc)
        
    elif para.imgtime == False:
        G = gpe.GPE(potfn = pot)
    # G = gpe.GPE()
    
    
elif para.omega != 0 and para.gamma == 0:
    from quTARANG.src.lib import rgpe
    G = rgpe.GPE(potfn = pot, wfcfn = wfc)
    # G = rgpe.GPE(omegafn = omega_)
    
elif para.omega == 0 and para.gamma != 0:
    raise ValueError('Dissipation has not implemented')  

else:
    raise ValueError('Both omega and gamma cannot be non-zero')    

evolution.time_advance(G)
