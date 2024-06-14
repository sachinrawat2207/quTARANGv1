#================================================================================
#                       Change the following parameters
#================================================================================
import sys
from pathlib import Path
from quTARANG.util import fns_util as util

real_dtype = 'float64'
complex_dtype = 'complex128'


pi = 3.141592653589793

# Device Setting 
device = 'gpu'            # Choose the device <'cpu'> to run on cpu and gpu to run on <'gpu'>
device_rank = 1      # Set GPU no in case if you are running on a single GPU else leave it as it is

# Set grid size 
Nx = 128
Ny = 128
Nz = 1

# Set box length
Lx = 20
Ly = 20
Lz = 1

# Set maximum time and dtquantum de
tmax = 1000        
dt = 0.001

# Choose the value of the non linerarity
g = 500

# Choose the chemical potential
mu = 0

# Choose the number of particles 
Npar = 1


type = 'a' # To enter in quantum droplet mode

init_usrdef = True

# Choose initial_condition setting in init_usd is False
initcond = 'cg_rot'

# If init_usrdef is True then set the input path or pass input through main
in_path =  '/home/sachin/quTARANG_v_1.6/output'
# print(in_path)
try:
    init_wfc = util.get_last_saved_file(in_path+'/wfc')
except:
    init_wfc = None
# print(init_wfc)
    
# init_wfc = 'wfc_t1828.280000.h5' #put the name of hdf5 file with extension
init_pot = 'pot.h5' #put the name of hdf5 file with extension

# Choose the scheme need to implement in the code
scheme = 'TSSP'            # Choose the shemes <'TSSP'>, <'RK4'> etc
imgtime = True          # set <False> for real time evolution and <True> for imaginary time evolution
delta = 1e-12

# To resume the Run
resume = False
overwrite = True
# rs_wfc = 'wfc_t3.9000000.h5'

# Wavefunction save setting
wfc_start_step = 0

# make wfc_iter too big to stop saving the wfc 
wfc_iter_step = 100

# Rms save setting. It Will not save for imaginary time!
save_rms = False
rms_start_step = 0
rms_iter_step = 10

# Energy save setting. It Will not save for imaginary time!
save_en = True
en_start_step = 0
en_iter_step = 100

# Printing iteration step
t_print_step = 10000

# Add rotation
omega = 0.5

# Add dissipation 
gamma = 0


# Set output folder path
if imgtime == True:
    op_path = '/home/sachin/quTARANG_v_1.6/outputgpelabtssp'

    save_en = False
    wfc_iter_step = 10000
    
elif imgtime == False:
    op_path = '/home/sachin/quTARANG_v_1.6/output'


#================================================================================
if Nx != 1 and Ny == 1 and Nz == 1:
    dimension = 1

elif Nx != 1 and Ny != 1 and Nz == 1:
    dimension = 2

elif Nx != 1 and Ny != 1 and Nz != 1:
    dimension = 3  
#================================================================================