from quTARANG import para

if para.device == 'gpu':
    import cupy as ncp
    import cupy.fft as fft
    dev = ncp.cuda.Device(para.device_rank)
    
    dev.use()
    

else:
    import numpy as ncp
    import pyfftw.interfaces.numpy_fft as fft 
        
        