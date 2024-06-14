from quTARANG import para
from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.univ import grid
from quTARANG.src.univ import data_io as IO
from quTARANG.config.config import ncp
import sys

#############################################################################################
#                   Numerical Schemes(without dissipation and rotation)             
#############################################################################################


###############################
# TSSP Scheme
###############################

def tssp_stepr(G, dt):
    return G.wfc * ncp.exp(-1j * (G.pot - para.mu + para.g  * ncp.abs(G.wfc)**2) * dt)

def tssp_stepk(G, dt):
    return G.wfck * ncp.exp(-0.5j * grid.ksqr * dt)
    
# For real time evolution
def time_adv_strang(G):
    G.wfc[:] = tssp_stepr(G, para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = tssp_stepk(G, para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = tssp_stepr(G, para.dt/2)

# For imaginary time evolution
def time_adv_istrang(G):
    G.wfc[:] = tssp_stepr(G, -1j * para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = tssp_stepk(G, -1j * para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = tssp_stepr(G, -1j * para.dt/2)
    if para.mu == 0:
        G.renorm()


#################################
# BESP Scheme imaginary time
#################################
def Lz(G):
    return - (grid.x_mesh * fft.inverse_transform1D(grid.ky_mesh * fft.forward_transform1D(G.U.temp1k, axis = 1), axis = 1) - \
        grid.y_mesh * fft.inverse_transform1D(grid.kx_mesh * fft.forward_transform1D(G.U.temp1k, axis = 0), axis = 0))


def BESP(G):
    #Need to test it
    G.renorm()
    G.U.temp[:] = G.pot + para.g * ncp.abs(G.wfc)**2
    alpha = 0.5 * (ncp.max(G.U.temp) + ncp.min(G.U.temp))
    G.U.temp1k[:] = G.wfc
    G.wfck[:] = 0
    while ncp.max(ncp.abs(G.wfck - G.U.temp1k)) > 1e-9:
        G.wfck[:] = G.U.temp1k[:]
        if para.omega == 0:
            G.U.temp1k[:] = (alpha - G.U.temp) * G.U.temp1k[:]
        else:
            G.U.temp1k[:] = (alpha - G.U.temp) * G.U.temp1k[:] + para.omega * Lz(G)
        G.U.temp1k[:] = 2/(2 + para.dt * (2 * alpha + grid.ksqr))*(fft.forward_transform(G.wfc + para.dt * G.U.temp1k))
        G.U.temp1k[:] = fft.inverse_transform(G.U.temp1k)
        G.U.temp1k[:] = G.U.temp1k[:]/G.norm_wfc(G.U.temp1k[:])
    G.wfc[:] = G.U.temp1k[:]
    G.renorm()
    

#################################
# TSSP Scheme quantum droplet
#################################

def etssp_stepr(G, dt):
    return G.wfc * ncp.exp(-1j * (G.pot - para.mu + para.g * ncp.nan_to_num(ncp.abs(G.wfc)**2 * ncp.log(ncp.abs(G.wfc)**2))) * dt)

def etssp_stepk(G, dt):
    return G.wfck * ncp.exp(-0.5j * grid.ksqr * dt)

def etime_adv_strang(G):
    G.wfc[:] = etssp_stepr(G, para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = etssp_stepk(G, para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = etssp_stepr(G, para.dt/2)

# For imaginary time evolution
def etime_adv_istrang(G):
    G.wfc[:] = etssp_stepr(G, -1j * para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = etssp_stepk(G, -1j * para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = etssp_stepr(G, -1j * para.dt/2)
    if para.mu == 0:
        G.renorm()


###############################
# RK4 Scheme
###############################  
# Need to check and validate RK4 scheme

def compute_RHS(G, psik):
    G.wfc[:] = fft.inverse_transform(psik)
    G.wfc[:] = -1j  * (grid.ksqr * psik/2 + fft.forward_transform((para.g * ncp.abs(G.wfc)**2 + G.pot) * G.wfc))
    return G.wfc

def time_adv_rk4(G):
    G.U.temp1k[:] = G.wfck + para.dt/2 * compute_RHS(G, G.wfck)
    G.U.temp2k[:] = G.wfck + para.dt/2 * compute_RHS(G, G.U.temp1k)
    G.U.temp3k[:] = G.wfck + para.dt * compute_RHS(G, G.U.temp2k)
    G.wfck[:] = G.wfck + para.dt/6 *(compute_RHS(G, G.wfck) + 2 * compute_RHS(G, G.U.temp1k) + 2 * compute_RHS(G, G.U.temp2k) + compute_RHS(G, G.U.temp3k))
    
    
#############################################################################################
#                          Numerical Schemes Rotation(for 2D and 3D)                                          
#############################################################################################

###############################
# Strang ADI TSSP scheme
###############################

def tssp_stepr_rot(G, dt):
    return G.wfc * ncp.exp(-1j * (G.pot + para.g  * (G.wfc * G.wfc.conj())) * dt)

def tssp_stepkx_rot(G, dt):
    return G.wfck * ncp.exp(-1j * (0.5 * grid.kx_mesh**2 + G.omega * grid.y_mesh * grid.kx_mesh) * dt)

def tssp_stepky_rot(G, dt):
    return G.wfck * ncp.exp(-1j * (0.5 * grid.ky_mesh**2 - G.omega * grid.x_mesh * grid.ky_mesh) * dt)

# For imaginary time evolution
def time_adv_istrang_rot(G):
    # Step1
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 0)    
    G.wfck[:] = tssp_stepkx_rot(G, -1j * para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 0)
    
    # Step2
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 1)
    G.wfck[:] = tssp_stepky_rot(G, -1j * para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 1)
    
    # Step3
    G.wfc[:] = tssp_stepr_rot(G, -1j * para.dt)
    
    # Step4
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 1)
    G.wfck[:] = tssp_stepky_rot(G, -1j * para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 1)
    
    # Step5
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 0)
    G.wfck[:] = tssp_stepkx_rot(G, -1j * para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 0)
    G.renorm()
    
# For real time evolution
def time_adv_strang_rot(G):
    # Step1
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 0)    
    G.wfck[:] = tssp_stepkx_rot(G, para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 0)
    
    # Step2
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 1)
    G.wfck[:] = tssp_stepky_rot(G, para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 1)
    
    # Step3
    G.wfc[:] = tssp_stepr_rot(G, para.dt)
    
    # Step4
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 1)
    G.wfck[:] = tssp_stepky_rot(G, para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 1)
    
    # Step5
    G.wfck[:] = fft.forward_transform1D(G.wfc, axis = 0)
    G.wfck[:] = tssp_stepkx_rot(G, para.dt/2)
    G.wfc[:] = fft.inverse_transform1D(G.wfck, axis = 0)


def set_scheme(G):
    global time_adv
    if para.scheme == 'TSSP'  and G.omega == 0:
        if para.imgtime == False:
            if para.type == 'qd':
                time_adv = etime_adv_strang 
            else:
                time_adv = time_adv_strang
        elif para.imgtime == True:
            if para.type == 'qd':
                time_adv = etime_adv_istrang 
            else:
                time_adv = time_adv_istrang
    
    if para.scheme == 'BESP':
        if para.imgtime == True:
            time_adv = BESP
        else:
            sys.exit("BESP is not implemented for real time.")
    elif para.scheme == 'TSSP' and G.omega != 0:
        if para.imgtime == False:
            time_adv = time_adv_strang_rot
        elif para.imgtime == True:
            time_adv = time_adv_istrang_rot
        
    elif para.scheme == 'RK4':
        time_adv = time_adv_rk4
    else:
        print("Please choose the correct scheme")
        quit()
    print("***** Scheme has been set *****")



def time_advance(G):
    print("***** Time advence started ***** ")
    t = grid.t_initial
    
    if para.scheme == "RK4":
        G.wfck[:] = fft.forward_transform(G.wfc)
        
    if para.imgtime == True:
        E = ncp.zeros(2)
        error = 1
        i=0
        G.renorm()
        E[0] = G.compute_te()
        while error > para.delta:
            if i%para.t_print_step == 0:
                IO.print_params(G, error = error, energy = E[1], iter = i)
            
            if(i >= para.wfc_start_step and (i - para.wfc_start_step)%para.wfc_iter_step == 0):
                IO.save_wfc(G, i*para.dt)
            time_adv(G)
            E[1] = G.compute_te()
            error = ncp.abs(E[0]-E[1])
            E[0] = E[1]
            i+=1
            
            t += para.dt
        IO.save_wfc(G, i*para.dt)
        IO.print_params(G, error = error, energy = E[1], iter = i)
    elif para.imgtime == False:
        for i in range(grid.nstep):
            
            if(i >= para.wfc_start_step and (i - para.wfc_start_step)%para.wfc_iter_step == 0):
                IO.save_wfc(G, t)
                
            if(para.save_en and i >= para.en_start_step and (i - para.en_start_step)%para.en_iter_step == 0):
                IO.compute_energy(G, t)

            if(para.save_rms and i >= para.rms_start_step and  (i - para.rms_start_step)%para.rms_iter_step == 0):
                IO.compute_rms(G, t)
            
            if i%para.t_print_step == 0:
                IO.print_params(G, time = t)
                # print(round(t,7), G.compute_te(), G.norm())
            
            if G.potfn != None:
                G.pot[:] = G.potfn(t)                    
            
            if G.omega != 0:
                if G.omegafn != None:
                    G.omega = G.omegafn(t)
            
            t += para.dt
            time_adv(G)
    
    if grid.nstep > 1:
        IO.save_wfc(G,t)
        if (para.save_en): 
            IO.compute_energy(G, t)
            IO.save_energy(G)
            
        if (para.save_rms):
            IO.compute_rms(G, t)
            IO.save_rms(G)
    
    print("***** Run Completed ***** ")