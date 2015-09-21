# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:13:20 2015

@author: hjalmar
"""
import string
from pycuda import compiler, gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class Softmax:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self.dtype_max = np.finfo(self.params['dtype']).max
        self._func = self._get_func()        
        
    def get(self, x_gpu, Ws_gpu, y_gpu, stream=None):

        blockDim_x = self.params['Ws_w']
        blockDim_y = self.params['C']

        if stream is None:
            stream = cuda.Stream()
            
        z_gpu = gpuarray.empty(self.params['C'], self.params['dtype'])            
                
        self._func[0](x_gpu, Ws_gpu, z_gpu, 
                      block=(blockDim_x, blockDim_y, 1),
                      stream=stream)

        self._func[1](self.dtype_max, z_gpu, y_gpu, 
                      block=(1, blockDim_y, 1),
                      stream=stream)
            
    def _get_func(self):
        """
        """

        kernel_template = """
            #include <math.h>
        
            #define C $C
            #define WS_W $WS_W
        
            __global__ void dot(double*x, double*Ws, double*z){
            
                __shared__ double sW[C][WS_W];
                
                const size_t tx = threadIdx.x;
                const size_t ty = threadIdx.y;
                
                if(ty < C && tx < WS_W){
                
                    sW[ty][tx] = Ws[ty * WS_W + tx];
                    z[ty] = 0.f;
                    
                    __syncthreads();
                    
                    for (int i = 0; i < WS_W; i++)
                        z[ty] += sW[ty][i] * x[i];                    
            
                }
            }


            __global__ void softmax(double dtype_max, double*z, double*y)
            {

                double s = 0.f;
            
                __shared__ double se[C];
                
                const size_t ty = threadIdx.y;

                if(ty < C){
                    
                    se[ty] = exp(z[ty]);

                    // TODO: Fix this
                    se[ty] = fmin(se[ty], dtype_max);
                        
                    __syncthreads();
                        
                    for (int i = 0; i < C; i++)
                        s += se[i];
                            
                    __syncthreads();
                
                    y[ty] = se[ty]/s;
                    
                    __syncthreads();
                }   
            }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(C = self.params['C'],
                                                 WS_W = self.params['Ws_w'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('dot'), module.get_function('softmax')


class Softmax_alt:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self.dtype_max = np.finfo(self.params['dtype']).max
        self._func = self._get_func()        
        
    def get(self, x_gpu, Ws_gpu, y_gpu, stream=None):

        blockDim_x = self.params['Ws_w']
        blockDim_y = self.params['C']
            
        if stream is None:
    
            self._func(self.dtype_max, x_gpu, Ws_gpu, y_gpu, 
                       block=(blockDim_x, blockDim_y, 1))
        else:
            
            self._func(self.dtype_max, x_gpu, Ws_gpu, y_gpu, 
                       block=(blockDim_x, blockDim_y, 1),
                       stream=stream)
            
    def _get_func(self):
        """
        """

        kernel_template = """
            #include <math.h>
        
            #define C $C
            #define WS_W $WS_W
                 
            __global__ void softmax(double dtype_max, double*x,
                                    double*Ws, double*y)
            {
        
                double s = 0.f;
                
                __shared__ double sW[C][WS_W];
                __shared__ double sz[C];    
                __shared__ double se[C];
                
                const size_t tx = threadIdx.x;
                const size_t ty = threadIdx.y;
                    
                if(ty < C && tx < WS_W){
                        
                    sW[ty][tx] = Ws[ty * WS_W + tx];
                    sz[ty] = 0.f;
                        
                    __syncthreads();
                    
                    for (int i = 0; i < WS_W; i++)
                        sz[ty] += sW[ty][i] * x[i];
                        
                    __syncthreads();
                    
                    se[ty] = exp(sz[ty]);
                    // TODO: Fix this
                    se[ty] = fmin(se[ty], dtype_max);
                        
                    __syncthreads();
                        
                    for (int i = 0; i < C; i++)
                        s += se[i];
                            
                    __syncthreads();
                
                    y[ty] = se[ty]/s;
                }   
            }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(C = self.params['C'],
                                                 WS_W = self.params['Ws_w'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('softmax')


###############################################################################


class DiffFromSoftmax:
    """
    
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, Ws_gpu, y_gpu, t_gpu, x_gpu, ds_gpu, stream=None):

        blockDim_x = self.params['C']
        blockDim_y = self.params['Ws_w']
            
        if stream is None:
    
            self._func(Ws_gpu, y_gpu, t_gpu, x_gpu, ds_gpu,
                       block=(blockDim_x, blockDim_y, 1))

        else:
            
            self._func(Ws_gpu, y_gpu, t_gpu, x_gpu, ds_gpu,
                       block=(blockDim_x, blockDim_y, 1),
                       stream=stream)

    
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define C $C
        #define WS_W $WS_W
        
        __global__ void d_from_softmax(double*Ws,
                                       double*y,
                                       double*t,
                                       double*x,
                                       double*ds)
        {       
               
            const size_t tx = threadIdx.x;
            const size_t ty = threadIdx.y;
           
            __shared__ double se[C];
            __shared__ double sw[WS_W][C];
            __shared__ double sx[WS_W];    
        
            if(tx < C && ty < WS_W){
    
                ds[ty] = 0.f;        
                se[tx] = y[tx] - t[tx];
                sw[ty][tx] = Ws[tx * WS_W + ty];
                sx[ty] = x[ty];
    
                __syncthreads();
    
                for(int i = 0; i < C; i++)
                    ds[ty] +=  sw[ty][i] * se[i];
    
                __syncthreads();
                
                ds[ty] *= (1 - sx[ty] * sx[ty]);
    
            }
        }
        """

        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(C = self.params['C'],
                                                 WS_W = self.params['Ws_w'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('d_from_softmax')
        

###############################################################################


class DiffToSoftmax:
    """
    
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, y_gpu, t_gpu, d_gpu, stream=None):

        blockDim_x = self.params['C']        

        if stream is None:
        
            self._func(y_gpu, t_gpu, d_gpu,
                       block=(blockDim_x, 1, 1))
        else:
            
            self._func(y_gpu, t_gpu, d_gpu,
                       block=(blockDim_x, 1, 1),
                       stream=stream)
        
                   
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define C $C
        
        __global__ void d_to_softmax(double*y, double*t, double*d)
        {       
           
            if (threadIdx.x < C){
                d[threadIdx.x] = y[threadIdx.x] - t[threadIdx.x];
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(C = self.params['C'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('d_to_softmax')


###############################################################################


class DiffAdd:
    """
    
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, ds_gpu, dp_gpu, stream=None):

        blockDim_x = self.params['w_d']        
        
        if stream is None:

            self._func(ds_gpu, dp_gpu,
                       block=(blockDim_x, 1, 1))
        else:

            self._func(ds_gpu, dp_gpu,
                       block=(blockDim_x, 1, 1),
                       stream=stream)


    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define W_W $W_W
        
        __global__ void d_add(double*ds, double*dp)
        {       
           
            if (threadIdx.x < W_W){
                ds[threadIdx.x] += dp[threadIdx.x];
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(W_W = self.params['W_w'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('d_add')


###############################################################################


class TNDiffDown:
    """
    
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    #def get(self, V_gpu, d_gpu, xcl_gpu, xcr_gpu, W_gpu, S_gpu,
     #       VVT_gpu, tmp_gpu, dp2_gpu, dpl_gpu, dpr_gpu, 
      #      stream=None):
                
    def get(self, V_gpu, d_gpu, xcl_gpu, xcr_gpu,
            W_gpu, dpl_gpu, dpr_gpu, stream=None):  
            
        S_gpu = gpuarray.empty(self.params['V_w'], self.params['dtype'])
        VVT_gpu = gpuarray.to_gpu(np.ndarray(shape=self.params['V_shape'],
                                             dtype=self.params['dtype'],
                                             strides=self.params['V_strides']))
        tmp_gpu = gpuarray.empty((self.params['V_w'], self.params['w_d']), 
                                  self.params['dtype'])
        dp2_gpu = gpuarray.empty(self.params['W_w'], self.params['dtype'])

        if stream is None:
            stream = cuda.Stream()

        gridDim_x = 2
        gridDim_y = 2
        gridDim_z = self.params['V_d']        
        blockDim_x = self.params['V_w']//gridDim_x
        blockDim_y = self.params['V_h']//gridDim_y

        self._func[0](V_gpu, VVT_gpu, 
                      block=(blockDim_x, blockDim_y, 1), 
                      grid=(gridDim_x, gridDim_y, gridDim_z),
                      stream=stream)
                   
        gridDim_x = 1
        gridDim_y = 2
        blockDim_x = self.params['V_d']
        blockDim_y = self.params['V_h']//gridDim_y

        self._func[1](VVT_gpu, d_gpu, xcl_gpu, xcr_gpu, tmp_gpu,
                      block=(blockDim_x, blockDim_y, 1), 
                      grid=(gridDim_x, gridDim_y, 1),
                      stream=stream)

        gridDim_x = 1
        gridDim_y = 2
        blockDim_x = self.params['V_d']
        blockDim_y = self.params['V_h']//gridDim_y

        self._func[2](tmp_gpu, S_gpu,
                      block=(blockDim_x, blockDim_y, 1), 
                      grid=(gridDim_x, gridDim_y, 1),
                      stream=stream)

        gridDim_x = 2
        gridDim_y = 1
        blockDim_x = self.params['V_w']//gridDim_x
        blockDim_y = self.params['V_d']

        self._func[3](W_gpu, d_gpu, S_gpu, xcl_gpu, xcr_gpu, dp2_gpu,
                      block=(blockDim_x, blockDim_y, 1), 
                      grid=(gridDim_x, gridDim_y, 1),
                      stream=stream)
                      
        blockDim_x = self.params['V_w']
        self._func[4](dp2_gpu, dpl_gpu, dpr_gpu,
                      block=(blockDim_x, 1, 1), 
                      grid=(1, 1, 1),
                      stream=stream)                     


    def _get_func(self):
        """
        """

        kernel_template = """
        
        #define V_W $V_W
        #define V_D $V_D

        __global__ void get_S0(double*V, double*tmp)
        {       
            
            // Thread position in each dimension
            const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
            const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
            const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
                
            if(tx < V_W && ty < V_W && tz < V_D){
            
                // tx, ty, tz index to flat index
                uint tid = tz * V_W * V_W + ty * V_W + tx;
                // Transposed flat index
                uint tidT = tz * V_W * V_W + tx * V_W + ty;
                
                tmp[tid] = V[tid] + V[tidT];
            }
        }


        __global__ void get_S1(double*VVT,
                               double*d,
                               double*xcl,
                               double*xcr,
                               double*tmp)
        {
        
            // Thread position in each dimension
            const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
            const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
            
            __shared__ double sxc2[V_W];

            if(tx < V_D && ty < V_W){

                tmp[ty * V_D + tx] = 0.f;

                sxc2[threadIdx.x] = xcl[threadIdx.x];
                sxc2[threadIdx.x+V_D] = xcr[threadIdx.x];
                    
                __syncthreads();
                        
                const size_t slice = tx * V_W * V_W;
                
                for (int i = 0; i < V_W; i++)
                    tmp[ty * V_D + tx] += VVT[slice + ty * V_W + i] * sxc2[i];

                __syncthreads();        
                
                tmp[ty * V_D + tx] *= d[tx];
            }    
        }


        __global__ void get_S2(double*tmp, double*S)
        {
            
            const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
            
            __shared__ double stmp [V_W][V_D];
        
            if(threadIdx.x < V_D && ty < V_W){
            
                stmp[ty][threadIdx.x] = tmp[ty * V_D + threadIdx.x];
                double sum = 0.f;     
                S[threadIdx.y] = 0.f;

                __syncthreads();
                
                for (int i = 0; i < V_D; i++)
                    sum += stmp[ty][i];
                S[ty] = sum;
            }
        }


        __global__ void TN_d_down(double*W, 
                                  double*d, 
                                  double*S, 
                                  double*xcl,
                                  double*xcr,
                                  double*dp2)
        {
                    
            const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
            const size_t bx = blockDim.x * blockIdx.x;

            __shared__ double sW[V_D][V_D];
            __shared__ double sd[V_D];
            __shared__ double sxc2[V_W];
                
            if (tx < V_W && ty < V_D){

                sxc2[threadIdx.x] = xcl[threadIdx.x];
                sxc2[threadIdx.x+V_D] = xcr[threadIdx.x];
                dp2[tx] = 0.f;
                sd[ty] = d[ty];
                
                // Transpose
                sW[threadIdx.y][threadIdx.x] = W[threadIdx.x * V_W + 
                                                 bx + threadIdx.y];
                    
                __syncthreads();
        
                for(int i = 0; i < V_D; i++)
                    dp2[threadIdx.y + bx] += sW[threadIdx.y][i] * sd[i];
        
                __syncthreads();
                
                dp2[tx] += S[tx];
                
                __syncthreads();
                
                dp2[tx] *= (1 - sxc2[tx] * sxc2[tx]);
                
            }
        }

        
        __global__ void split_dp2(double*dp2, double*dpl, double*dpr){
        
            const size_t tx = threadIdx.x;
            
            if (tx < V_D)
                dpl[tx] = dp2[tx];
            if (tx >= V_D && tx < V_W)
                dpr[tx - V_D] = dp2[tx];
        
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(V_W = self.params['V_w'],
                                                 V_D = self.params['V_d'])
        module = compiler.SourceModule(kernel_code)
        get_S0 = module.get_function('get_S0')
        get_S1 = module.get_function('get_S1')
        get_S2 = module.get_function('get_S2')
        TN_d_down = module.get_function('TN_d_down')
        split_dp2 = module.get_function('split_dp2')
        return get_S0, get_S1, get_S2, TN_d_down, split_dp2


###############################################################################


class TNActivation:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, V_gpu, xcl_gpu, xcr_gpu, W_gpu, x_gpu, stream=None):
        """
        """
        if stream is None:
            stream = cuda.Stream()

        # Temporary variables
        z_gpu = gpuarray.empty((self.params['V_d'], 
                                self.params['V_w']), self.params['dtype'])
        xc2_gpu = gpuarray.empty(2*self.params['w_d'], self.params['dtype'])
            
        blockDim_x = self.params['V_w']
        
        self._func[0](xcl_gpu, xcr_gpu, xc2_gpu,
                      block=(blockDim_x, 1, 1),
                      stream=stream)            
        
        gridDim_z = self.params['V_d']
        blockDim_y = self.params['V_w']
        
        self._func[1](V_gpu, xc2_gpu, z_gpu,
                      block=(1, blockDim_y, 1),
                      grid=(1, 1, gridDim_z),
                      stream=stream)

        blockDim_y = self.params['W_h']
        
        self._func[2](W_gpu, xc2_gpu, z_gpu, x_gpu,
                      block=(1, blockDim_y, 1),
                      grid=(1, 1, 1),
                      stream=stream)                   
                   
    
    def _get_func(self):
        """   
        """    

        kernel_template = """

        #include <math.h>

        #define V_W $V_W
        #define V_D $V_D
        #define W_H $W_H


        __global__ void merge_xc(double*xcl, double*xcr, double*xc2){

            if (threadIdx.x < W_H)
                xc2[threadIdx.x] = xcl[threadIdx.x];

            if (threadIdx.x >= W_H && threadIdx.x < V_W)
                xc2[threadIdx.x] = xcr[threadIdx.x - W_H];
        }

        __global__ void TN_activation0(double*V, double*xc2, double*z)
        {
        
            const size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
            const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
            
            if (ty < V_W && tz < V_D){
            
                z[tz * V_W + ty] = 0.f;
                
                __syncthreads();
                    
                const size_t slice = tz * V_W * V_W;
                
                for (int i = 0; i < V_W; i++)
                    z[tz * V_W + ty] += V[slice + ty * V_W + i] * xc2[i];
            }
        }
        
        __global__ void TN_activation1(double*W, double*xc2, 
                                       double*z, double*x){
                    
            const size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
                    
            if (ty < W_H){
            
                x[ty] = 0.f;
                
                __syncthreads();
                    
                for(int i = 0; i < V_W; i++)
                    x[ty] += xc2[i] * (W[ty * V_W + i] + z[ty * V_W + i]);
        
                __syncthreads();
            
                x[ty] = tanh(x[ty]);
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(V_W = self.params['V_w'],
                                                 V_D = self.params['V_d'],
                                                 W_H = self.params['W_h'])
        module = compiler.SourceModule(kernel_code)
        merge_xc = module.get_function('merge_xc')
        TN_activation0 = module.get_function('TN_activation0')
        TN_activation1 = module.get_function('TN_activation1')
        return merge_xc, TN_activation0, TN_activation1


###############################################################################    
       

class DiffFromNode:
    """
    
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, xcl_gpu, xcr_gpu, d_gpu, 
            W_gpu, dpl_gpu, dpr_gpu, stream=None):
        """
        """
                
        gridDim_x = 2
        blockDim_x = self.params['W_w']//gridDim_x
        blockDim_y = self.params['W_h']
        
        if stream is None:
        
            self._func(xcl_gpu, xcr_gpu, d_gpu, W_gpu, dpl_gpu, dpr_gpu,
                       block=(blockDim_x, blockDim_y, 1),
                       grid=(gridDim_x, 1, 1))
        
        else:

            self._func(xcl_gpu, xcr_gpu, d_gpu, W_gpu, dpl_gpu, dpr_gpu,
                       block=(blockDim_x, blockDim_y, 1),
                       grid=(gridDim_x, 1, 1),
                       stream=stream)
                   
    
    def _get_func(self):
        """
        """    

        kernel_template = """
        
        #define W_W $W_W
        #define W_H $W_H
            
        __global__ void d_from_node(double*xcl,
                                    double*xcr,
                                    double*d,
                                    double*W,
                                    double*dpl,
                                    double*dpr){
            
            const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
            const size_t bx = blockDim.x * blockIdx.x;
            
            __shared__ double sW[W_H][W_H];
            __shared__ double sd[W_H];
            __shared__ double sxc2[W_W];
            __shared__ double sdp2[W_W];
                
            if (tx < W_W && ty < W_H){

                sdp2[tx] = 0.f;
                
                sxc2[threadIdx.x] = xcl[threadIdx.x];
                sxc2[threadIdx.x + W_H] = xcr[threadIdx.x];

                sd[ty] = d[ty];                
                // Transpose
                sW[threadIdx.y][threadIdx.x] = W[threadIdx.x* W_W + 
                                                 bx + threadIdx.y];
                    
                __syncthreads();
        
                for(int i = 0; i < W_H; i++){
                    sdp2[threadIdx.y + bx] += sW[threadIdx.y][i] * sd[i];
                }
        
                __syncthreads();
                
                sdp2[tx] *= (1 - sxc2[tx] * sxc2[tx]);
                
                __syncthreads();
                
                if (tx < W_H)
                    dpl[threadIdx.x] = sdp2[tx];
                if (tx >= W_H)
                    dpr[threadIdx.x] = sdp2[tx];                
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(W_W = self.params['W_w'],
                                                 W_H = self.params['W_h'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('d_from_node')


###############################################################################


class GradWs:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, d_gpu, x_gpu, GWs_gpu, stream=None):
        
        blockDim_x = self.params['Ws_w']
        blockDim_y = self.params['Ws_h']
        
        if stream is None:        
        
            self._func(d_gpu, x_gpu, GWs_gpu,
                       block=(blockDim_x, blockDim_y, 1))
                      
        else:

            self._func(d_gpu, x_gpu, GWs_gpu,
                       block=(blockDim_x, blockDim_y, 1),
                       stream=stream)                      
                   
    
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define Ws_W $Ws_W
        #define Ws_H $Ws_H
        
        __global__ void grad_Ws(double*d, double*x, double*grad){
        
            const size_t tx = threadIdx.x;
            const size_t ty = threadIdx.y;
            
            __shared__ double sd[Ws_H];
            __shared__ double sx[Ws_W];
            
            if (tx < Ws_W && ty < Ws_H){
            
                sx[threadIdx.x] = x[threadIdx.x];
                sd[threadIdx.y] = d[threadIdx.y];
                
                __syncthreads();

                grad[ty * Ws_W + tx] += sd[ty] * sx[tx];
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(Ws_W = self.params['Ws_w'],
                                                 Ws_H = self.params['Ws_h'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('grad_Ws')


###############################################################################


class GradL:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, word_idx, ds_gpu, d_gpu, GL_gpu, stream=None):
        
        blockDim_x = self.params['w_d']
        
        if stream is None:        
        
            self._func(word_idx, ds_gpu, d_gpu, GL_gpu,
                       block=(blockDim_x, 1, 1))
                      
        else:

            self._func(word_idx, ds_gpu, d_gpu, GL_gpu,
                       block=(blockDim_x, 1, 1),
                       stream=stream)                      
                   
    
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define D $D
        
        __global__ void grad_L(int row, double*ds, double*d, double*grad){
        
            if (threadIdx.x < D){
            
                grad[row * D + threadIdx.x] += (ds[threadIdx.x] + 
                                                d[threadIdx.x]);

            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(D = self.params['w_d'])
        module = compiler.SourceModule(kernel_code)
        return module.get_function('grad_L')


###############################################################################


class GradV:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()

    def get(self, xcl_gpu, xcr_gpu, d_gpu, GV_gpu, stream=None):
        
        if stream is None:
            stream = cuda.Stream()

        xc2_gpu = gpuarray.empty(2*self.params['w_d'], self.params['dtype'])
        blockDim_x = self.params['V_w']
        
        self._func[0](xcl_gpu, xcr_gpu, xc2_gpu,
                      block=(blockDim_x, 1, 1),
                      stream=stream)
        
        gridDim_x = 2
        gridDim_y = 2
        blockDim_x = self.params['V_w']//gridDim_x
        blockDim_y = self.params['V_h']//gridDim_y
        
        self._func[1](xc2_gpu, d_gpu, GV_gpu,
                      block=(blockDim_x, blockDim_y, 1),
                      grid=(gridDim_x, gridDim_y, 1),
                      stream=stream)
                      
    
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define V_W $V_W
        #define V_D $V_D
        
        __global__ void merge_xc(double*xcl, double*xcr, double*xc2){
                    
            if (threadIdx.x < V_D)
                xc2[threadIdx.x] = xcl[threadIdx.x];

            if (threadIdx.x >= V_D && threadIdx.x < V_W)
                xc2[threadIdx.x] = xcr[threadIdx.x - V_D];
        }
        
        
        __global__ void TN_dV(double*xc2, double*d, double*dV){
        
            const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (tx < V_W && ty < V_W){
            
                for (int k = 0; k < V_D; k++){

                    const size_t slice = k * V_W * V_W;
                    dV[slice + ty * V_W + tx] += d[k] * xc2[ty] * xc2[tx];
                    
                    __syncthreads();
                }
            }
        }
        
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(V_W = self.params['V_w'],
                                                 V_D = self.params['V_d'])
        module = compiler.SourceModule(kernel_code)
        merge_xc = module.get_function('merge_xc')
        TN_dV = module.get_function('TN_dV')
        return merge_xc, TN_dV


###############################################################################


class GradW:
    """
    """    
    def __init__(self, params):
        
        self.params = params
        self._func = self._get_func()


    def get(self, d_gpu, xcl_gpu, xcr_gpu, GW_gpu, stream=None):
        
        if stream is None:
            stream = cuda.Stream()
        
        xc2_gpu = gpuarray.empty(2*self.params['w_d'], self.params['dtype'])

        blockDim_x = self.params['W_w']
        
        self._func[0](xcl_gpu, xcr_gpu, xc2_gpu,
                      block=(blockDim_x, 1, 1), stream=stream)
                    
        gridDim_x = 2
        blockDim_x = self.params['W_w']//gridDim_x
        blockDim_y = self.params['W_h']
        
        self._func[1](d_gpu, xc2_gpu, GW_gpu,
                      block=(blockDim_x, blockDim_y, 1),
                      grid=(gridDim_x, 1, 1), stream=stream)
                   
    
    def _get_func(self):
        """
    
        """    

        kernel_template = """
        
        #define W_W $W_W
        #define W_H $W_H
        
        __global__ void merge_xc(double*xcl, double*xcr, double*xc2){
                    
            if (threadIdx.x < W_H)
                xc2[threadIdx.x] = xcl[threadIdx.x];

            if (threadIdx.x >= W_H && threadIdx.x < W_W)
                xc2[threadIdx.x] = xcr[threadIdx.x - W_H];
        }
        
        
        __global__ void grad_W(double*d, double*xc2, double*grad){
        
            const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t ty = threadIdx.y;
            
            __shared__ double sd[W_H];
            __shared__ double sxc2[W_W];
            
            if (tx < W_W && ty < W_H){
            
                sd[ty] = d[ty];
                sxc2[tx] = xc2[tx];
            
                __syncthreads();

                grad[ty * W_W + tx] += sd[ty] * sxc2[tx];
            }
        }
        """
        kernel_template = string.Template(kernel_template)
        kernel_code = kernel_template.substitute(W_W = self.params['W_w'],
                                                 W_H = self.params['W_h'])
        module = compiler.SourceModule(kernel_code)
        merge_xc = module.get_function('merge_xc')
        grad_W = module.get_function('grad_W')
        return merge_xc, grad_W


###############################################################################                                 