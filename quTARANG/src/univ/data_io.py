from quTARANG.config.config import ncp
from quTARANG import para  
from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.univ import grid
import h5py as hp
import os
from pathlib import Path
from quTARANG.util import fns_util as util

if para.dimension == 2 and para.init_usrdef == False:
    from quTARANG.initial_cond import dimension2 as dim

elif para.dimension == 3 and para.init_usrdef == False:
    from quTARANG.initial_cond import dimension3 as dim

def print_params(G, time=0, iter=0, error = 0, energy = 0):
    # print("Chemical Potential: ", G.compute_chmpot())
    if para.imgtime == True:
        print("Iteration: ", iter)
        print("Stoping criteria for energy: delta E<", para.delta)
        print('delta E: ', error)
    
    
    elif para.imgtime == False:
        print('Time step: ', round(time, 7))
    print('Chemical Potential: ', G.compute_chmpot())
    print('Energy: ', G.compute_te())
    print('Particle Number: ', G.norm())
    print('Lz: ', G.compute_Lz())
    if para.omega != 0:
        print('omega: ', G.omega)
        
        if para.imgtime == True:
            print('xrms: ', G.compute_xrms())
            print('yrms: ', G.compute_yrms())
            
    print("-----------------------------------------\n")
    
def show_params():
    print('[Nx, Ny, Nz]: ', para.Nx, para.Ny, para.Nz)
    print('[Lx, Ly, Lz]:', para.Lx, para.Ly, para.Lz)
    print('dimension: ', para.dimension)
    print("g: ", para.g)
    print('Scheme:', para.scheme)
    print('Imaginary time:', para.imgtime)
    print("Npar: ", para.Npar)
    print("omega: ", para.omega)
    print('dt: ', para.dt)
    
    if para.init_usrdef:
        print("inital condition:", para.init_usrdef)
    if para.imgtime == True:
        print("Stoping criteria for energy: delta E<", para.delta)
    
    elif para.imgtime == False:
        print('Initial time: ', grid.t_initial)
        print('Final time: ', para.tmax)
        print('Total iterations:', grid.nstep)
    if para.resume == True:
        print('RUN Resumed!')
    print('')
    
# directory generation
def gen_path():
    if not Path(para.op_path).exists():
        os.makedirs(para.op_path)
    if not (Path(para.op_path)/'wfc').exists():   
        os.mkdir(Path(para.op_path)/'wfc')


def set_initcond(G):        
    if para.init_usrdef == True:
        if G.wfcfn !=  None:
            G.wfc[:] = G.wfcfn()
            
        else:
            f1 = hp.File(Path(para.in_path)/'wfc'/para.init_wfc, 'r')
            if para.device == 'gpu':
                G.wfc[:] = ncp.asarray(f1['wfc'])
            else:
                G.wfc[:] = f1['wfc']
            f1.close()
            
        if G.potfn != None:
            G.pot = G.potfn(0)
        else:
            f2 = hp.File(Path(para.in_path)/para.init_pot, 'r')
            if para.device == 'gpu':
                G.pot[:] = (ncp.asarray(f2['pot'])).real
            else:
                G.pot[:] = (f2['pot']).real
            f2.close()
            
    else:
        if para.initcond == "srp":
            G.wfc[:], G.pot[:] = dim.srp()
        
        elif para.initcond == "test":
            G.wfc[:], G.pot[:] = dim.test()
        
        elif para.initcond == 'cg_rot':
            G.wfc[:], G.pot[:] = dim.rot_centralgaussian()
        
        elif para.initcond == 'adhikari_rot':
            G.wfc[:], G.pot[:] = dim.rot_centralgaussianadhi()
        save_wfc(G,0)
    print('***** Wavefunction and potential initializes *****')
    
def set_resume_initcond(G): 
    f1 = hp.File(Path(para.op_path)/'wfc'/grid.rs_wfc, 'r')
    if G.potfn != None:
        G.pot[:] = G.potfn(grid.t_initial)
    else:
        f2 = hp.File(Path(para.op_path)/'pot.h5', 'r')
        if para.device == 'gpu':
            G.pot[:] = (ncp.asarray(f2['pot'])).real
        else:
            G.pot[:] = (f2['pot']).real
        f2.close()
        
    if para.device == 'gpu':
        G.wfc[:] = ncp.asarray(f1['wfc'])
    else:
        G.wfc[:] = f1['wfc']
    f1.close()
    
def save_wfc(G, t):
    if para.scheme == 'RK4':
        G.wfc = fft.inverse_transform(G.wfck)
    f1 = hp.File( Path(para.op_path)/('wfc/' + 'wfc_t%1.6f.h5'%t), 'w')
    if para.device == 'gpu':
        f1.create_dataset('wfc', data = ncp.asnumpy(G.wfc))
    elif para.device == 'cpu':
        f1.create_dataset('wfc', data = G.wfc)
    f1.close()
    
    if t==0:
        f2 = hp.File( Path(para.op_path)/'pot.h5', 'w')
        if para.device == 'gpu':
            f2.create_dataset('pot', data = ncp.asnumpy(G.pot))
            
        elif para.device == 'cpu':
            f2.create_dataset('pot', data = G.pot)
        f2.close()

def compute_rms(G, t):
    if para.scheme == 'RK4':
        G.wfc = fft.inverse_transform(G.wfck)
        
    if para.device == 'gpu':            
        if para.dimension >= 1:
            G.xrms.append(ncp.asnumpy(G.compute_xrms()))
            G.srmstime.append(ncp.asnumpy(t))
        
        if para.dimension >= 2:    
            G.yrms.append(ncp.asnumpy(G.compute_yrms()))
            G.rrms.append(ncp.asnumpy(G.compute_rrms()))
        
        if para.dimension == 3:
            G.zrms.append(ncp.asnumpy(G.compute_zrms()))
        
    elif para.device == 'cpu':
        if para.dimension >= 1:
            
            G.xrms.append(G.compute_xrms())
            G.srmstime.append(t)

        if para.dimension >= 2: 
            G.yrms.append(G.compute_yrms())
            G.rrms.append(G.compute_rrms())
        
        if para.dimension == 3:
            G.zrms.append(G.compute_zrms())

        
def save_rms(G):
    filename = util.new_filename(Path(para.op_path)/'rms.h5')
    f = hp.f = hp.File(filename, 'w')
    if para.dimension >= 1:
        f.create_dataset('xrms', data = G.xrms)
        f.create_dataset('t', data = G.srmstime)
    
    if para.dimension >= 2:
        f.create_dataset('yrms', data = G.yrms)
        f.create_dataset('rrms', data = G.rrms)
        
    if para.dimension == 3:
        f.create_dataset('zrms', data = G.zrms)
    f.close()


def compute_energy(G, t):
    if para.device == 'gpu':
        G.te.append(ncp.asnumpy(G.compute_te()))
        G.chmpot.append(ncp.asnumpy(G.compute_chmpot()))
        if G.omega == 0:
            ckec, ckei = G.ke_dec()
            G.ke.append(ncp.asnumpy(G.compute_ke()))
            G.kei.append(ncp.asnumpy(ckei))
            G.kec.append(ncp.asnumpy(ckec))
            G.qe.append(ncp.asnumpy(G.compute_qe()))
            G.ie.append(ncp.asnumpy(G.compute_ie()))
            G.pe.append(ncp.asnumpy(G.compute_pe()))
        if G.omega != 0:
            G.Lz.append(ncp.asnumpy(G.compute_Lz()))
    elif para.device == 'cpu':
        G.te.append(G.compute_chmpot())
        G.chmpot.append(G.compute_te())
        if G.omega != 0:
            G.Lz.append(G.compute_Lz())
        else:
            ckec, ckei = G.ke_dec()
            G.ke.append(G.compute_ke())
            G.kei.append(ckei)
            G.kec.append(ckec)
            G.qe.append(G.compute_qe())
            G.ie.append(G.compute_ie())
            G.pe.append(G.compute_pe()) 
        
    G.estime.append(t)
    
# Function to save energy in file
def save_energy(G):
    filename = util.new_filename(Path(para.op_path)/'energies.h5')
    f = hp.File(filename, 'w')
    f.create_dataset('tenergy', data = G.te)
    f.create_dataset('t', data = G.estime)
    
    if para.omega != 0:
        filename1 = util.new_filename(Path(para.op_path)/'Lz.h5') 
        f1 = hp.File(filename1, 'w')
        f1.create_dataset('Lz', data = G.Lz)
        f1.create_dataset('mu', data = G.chmpot)
        f1.create_dataset('t', data = G.estime)
        f1.close()
    else:
        f.create_dataset('kec', data = G.kec)
        f.create_dataset('kei', data = G.kei)
        f.create_dataset('ie', data = G.ie)
        f.create_dataset('qe', data = G.qe)
        f.create_dataset('pe', data = G.pe)
        f.create_dataset('ke', data = G.ke)
        f.close()
