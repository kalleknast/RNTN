# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:59:04 2015

@author: hjalmar
"""
import numpy as np
from RNTN_utils import get_proto_tree, read_proto_trees, load_vocabulary
from pycuda import compiler, gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from copy import deepcopy
from RNTN_cuda import Softmax, TNActivation        # Forward routines
from RNTN_cuda import TNDiffDown, DiffFromSoftmax  # Error
from RNTN_cuda import DiffAdd, DiffToSoftmax       # Error
from RNTN_cuda import GradWs, GradV, GradW, GradL  # Gradients

params = {'BatchSz': 25,        # Mini-batch size
          'eta': 0.01,          # Learnning rate
          'lambda': 0.01,       # Regularization parameter
          'C': 5,               # Number of classes
          'w_d': 25,            # length of word vector
          'Ws_w': 25,           # Ws width
          'Ws_h': 5,            # Ws height
          'V_w': 50,            # V width
          'V_h': 50,            # V height
          'V_d': 25,            # V depth
          'W_w': 50,            # W width
          'W_h': 25,            # W height
          'V_strides': [0, 0, 0],
          'V_shape': [0, 0, 0],
          'dtype': np.float64,
          }

fn_vocab = './data/vocabulary.txt'
fn_train = './data/train.txt'
# Vocabulary length
vocabulary = load_vocabulary(fn_vocab)
n_words = len(vocabulary)

dtype = params['dtype']
nBytes = np.dtype(dtype).itemsize
shape = (params['V_h'], params['V_w'], params['V_d'])
strides = (shape[1] * nBytes, nBytes, shape[0] * shape[1] * nBytes)                                        
params['V_strides'] = strides
params['V_shape'] = shape
r = 0.0001

# Sentiment classification matrix (paper page 4).
Ws = np.random.uniform(-r, r, size=(params['Ws_h'],
                                    params['Ws_w'])).astype(dtype)
# initialize weights for parent computation (paper, page 4).
W = np.random.uniform(-r, r, size=(params['W_h'], params['W_w'])).astype(dtype)
# Initialize word vectors and store in the word embedding matrix, L.
# L will be trained together with the other parameters.
L = np.random.uniform(-r, r, size=(n_words, params['w_d'])).astype(dtype)
# Tensor
nBytes = np.dtype(dtype).itemsize
shape = (params['V_h'], params['V_w'], params['V_d'])
buffer = np.random.uniform(-r, r, size=shape)
# Fix the strides so that the 3D dim changes last -> easier transpose (for me)
strides = (shape[1] * nBytes, nBytes, shape[0] * shape[1] * nBytes)
V = np.ndarray(shape=shape, buffer=buffer, dtype=dtype, strides=strides)

GWs = np.zeros(Ws.shape, dtype=dtype)
GW = np.zeros(W.shape, dtype=dtype)
GV = np.ndarray(shape=shape, buffer=np.zeros(shape),
                dtype=dtype, strides=strides)
GL = np.zeros(L.shape, dtype=dtype)
    
Counters = {'W': 0,
            'Ws': 0,
            'V': 0,
            'L': np.zeros(L.shape[0])}

Ws_gpu = gpuarray.to_gpu(Ws)
W_gpu = gpuarray.to_gpu(W)
V_gpu = gpuarray.to_gpu(V)

GWs_gpu = gpuarray.to_gpu(GWs)
GW_gpu = gpuarray.to_gpu(GW)
GV_gpu = gpuarray.to_gpu(GV)
GL_gpu = gpuarray.to_gpu(GL)


########################
# Compile the functions
#######################
# Forward
classification = Softmax(params)
activation = TNActivation(params)
# Backward error
error_down = TNDiffDown(params)
error_from_softmax = DiffFromSoftmax(params)
error_to_softmax = DiffToSoftmax(params)
error_complete = DiffAdd(params)
# Backward gradient
grad_Ws = GradWs(params)
grad_V = GradV(params)
grad_W = GradW(params)
grad_L = GradL(params)


def get_tree(proto_tree, N, vocabulary, L, params):
    '''
    '''

    dtype = [('node_idx', int),
             ('pair', int),
             ('side', '|S1'),
             ('parent', int),
             ('child_left', int),
             ('child_right', int),
             ('depth', int),
             ('leaf', bool),
             ('t', np.object),
             ('t_gpu', np.object),
             ('y_gpu', np.object),
             ('x_gpu', np.object),
             ('d_gpu', np.object),
             ('ds_gpu', np.object),
             ('d2s_gpu', np.object),
             ('word_idx', int),
             ('word', 'U27')]

    tree = np.recarray(N, dtype=dtype)
    tree[:] = -1
    tree['pair'] = -1
    
    children_stack = []
    node_idx = 0
    side = ''
    depth = 1
    
    #-------------- Add data to tree -----------------#
    def add_data(wrd, leaf, child_l=None, child_r=None):
    
        tree[node_idx]['node_idx'] = node_idx
        tree[node_idx]['side'] = side
        tree[node_idx]['t'] = np.zeros(params['C'], dtype=np.float64)
        tree[node_idx]['t'][wrd[0]] = 1.0
        tree[node_idx]['depth'] = depth
        tree[node_idx]['leaf'] = leaf
        if leaf:
            tree[node_idx]['word'] = wrd[1]
            tree[node_idx]['word_idx'] = vocabulary.index(wrd[1])
            x_gpu = gpuarray.to_gpu(L[tree[node_idx]['word_idx'], :])
            tree[node_idx]['x_gpu'] = x_gpu
        else:
            tree[node_idx]['word'] = ''
            tree[node_idx].child_right = child_r
            tree[node_idx].child_left = child_l
            # Add parent
            tree[child_r]['parent'] = tree[node_idx]['node_idx']
            tree[child_l]['parent'] = tree[node_idx]['node_idx']            
            tree[node_idx]['x_gpu'] = gpuarray.empty(params['w_d'], np.float64)
        
        tree[node_idx]['d_gpu'] = gpuarray.empty(params['w_d'], np.float64)
        tree[node_idx]['ds_gpu'] = gpuarray.empty(params['w_d'], np.float64)
        tree[node_idx]['d2s_gpu'] = gpuarray.empty(params['C'], np.float64)
        tree[node_idx]['y_gpu'] = gpuarray.empty(params['C'], np.float64)
        tree[node_idx]['t_gpu'] = gpuarray.to_gpu(tree[node_idx]['t'])
        # All pairs at current depth
        pairs = tree['pair'][tree['depth'] == depth]
        # Max is last pair
        if pairs.max() == -1:  # No pairs at this depth
            tree[node_idx]['pair'] = 1  # Start new pair
        elif (pairs > -1).sum() % 2: # Odd number -> complete the last pair               
            tree[node_idx]['pair'] = pairs.max()
        else:  # Left half of new pair
            tree[node_idx]['pair'] = pairs.max() + 1 
    #-----------------------------------------------#

    pos = 'proto_tree'
    while len(proto_tree) > 1:
        if len(eval(pos)) == 3: # Both branches
            pos += '[1]' # go down in left
            depth += 1
            side = 'l' # left side

        elif len(eval(pos)) == 2: # Only right branch or leaf

            if type(eval(pos + '[1]')) is str:  # leaf
                # Cut
                pos = pos[:-3]
                wrd = eval(pos).pop(1)
                add_data(wrd, True)
                # Put node_idx in the children_stack, but don't pop since its
                # a leaf node, ie no children
                children_stack.append(node_idx)
                node_idx += 1
                # Climb up
                depth -= 1

            else:  # right branch
                pos += '[1]' # go down in right
                depth += 1
                side = 'r' # right side

        elif len(eval(pos)) == 1: # Branch node, w both children cut
            # Cut
            pos = pos[:-3]
            wrd = eval(pos).pop(1)
            # left OR right side in a pair?
            if len(eval(pos)) == 2:         # if parent len is 2 -> left
                side = 'l' # left side
            elif len(eval(pos)) == 1:       # if parent len is 1 -> right
                side = 'r' # right side

            # Pop children
            child_r = children_stack.pop()
            child_l = children_stack.pop()
            add_data(wrd, False, child_l=child_l, child_r=child_r)
            # Put node_idx in the children_stack
            children_stack.append(node_idx)
            node_idx += 1
            # Climb up
            depth -= 1

    side = ''
    child_r = children_stack.pop()
    child_l = children_stack.pop()
    add_data(proto_tree, False, child_l=child_l, child_r=child_r)
    tree[node_idx]['parent'] = -1  # fix -1/last element mix up.
    
        
    return tree
   

def get_cost(T, Y, THETA, lmbd):
    """
    Eq. 2 in paper.
    """
    E = -((T * np.log(Y)).sum()) + lmbd * np.dot(THETA.T, THETA)
    return E


def forward(tree):
    """
    """

    # Depth 1 is top node, max depth is at some leaf node
    depths = np.unique(tree.depth)[::-1]      # Start at max depth
    
    for level in depths:
        level_idx = (tree.depth == level).nonzero()[0]
        streams = []

        for j, node_idx in enumerate(level_idx):
            
            node = tree[node_idx]
            streams.append(cuda.Stream())
                            
            if not node.leaf:   
                #---------------- Hidden & top specific -------------#
                xcl_gpu = tree[node.child_left].x_gpu
                xcr_gpu = tree[node.child_right].x_gpu

                activation.get(V_gpu, xcl_gpu, xcr_gpu, W_gpu, 
                               node.x_gpu, stream=streams[-1])
                #----------------------------------------------------#                               

            #-------- Same for both leaf & hidden nodes ---------#
            classification.get(node.x_gpu, Ws_gpu,
                               node.y_gpu, stream=streams[-1])
            #----------------------------------------------------#
        
            # Nvidia compute capability 5: 
            # max number of concurrent kernels is 32
            if not (j+1)%32:
                # Synchronize before launching new kernels
                cuda.Context.synchronize()
                streams = []

        # wait for all streams to complete before 
        # moving to next level in the tree
        cuda.Context.synchronize()


def backward(tree, Counters):
    """
    """
    
    # Depth 1 is top node, max depth is at some leaf node
    depths = np.unique(tree.depth)

    for level in depths:
        level_idx = (tree.depth == level).nonzero()[0]
        streams = []

        for j, node_idx in enumerate(level_idx):

            node = tree[node_idx]
            streams.append(cuda.Stream())
            
            #-------- Same for both leaf & hidden nodes ---------#
            error_to_softmax.get(node.y_gpu, node.t_gpu, 
                                 node.d2s_gpu, stream=streams[-1])
            grad_Ws.get(node.d2s_gpu, node.x_gpu, GWs_gpu, stream=streams[-1])
            Counters['Ws'] += 1
            error_from_softmax.get(Ws_gpu, node.y_gpu, node.t_gpu,
                                   node.x_gpu, node.ds_gpu, stream=streams[-1])
            grad_Ws.get(node.d2s_gpu, node.x_gpu, GWs_gpu, stream=streams[-1])
            Counters['Ws'] += 1
            dp_gpu = tree[node.parent].d_gpu
            #----------------------------------------------------#

            if node.leaf:
                #----------------- Leaf specific ----------------#
                grad_L.get(node.word_idx, node.ds_gpu,
                           dp_gpu, GL_gpu, stream=streams[-1])
                Counters['L'][node.word_idx] += 1            
                #------------------------------------------------#            
                
            else:
                #-------------- Hidden/top specific -------------#
                child_l = tree[node.child_left]
                child_r = tree[node.child_right]
                # Errors
                error_down.get(V_gpu, node.ds_gpu, child_l.x_gpu,
                               child_r.x_gpu, W_gpu, child_l.d_gpu, 
                               child_r.d_gpu, stream=streams[-1])
                if node.depth > 1:  # No parent error for top node
                    error_complete.get(node.ds_gpu, dp_gpu, stream=streams[-1])
                # Gradients
                grad_W.get(node.ds_gpu, child_l.x_gpu, child_r.x_gpu,
                           GW_gpu, stream=streams[-1])
                Counters['W'] += 1
                grad_V.get(child_l.x_gpu, child_r.x_gpu, node.ds_gpu,
                           GV_gpu, stream=streams[-1])
                Counters['V'] += 1                 
                #------------------------------------------------#
            
            # Nvidia compute capability 5: 
            # max number of concurrent kernels is 32
            if not (j+1)%32:
                # Synchronize before launching new kernels
                cuda.Context.synchronize()
                streams = []

        cuda.Context.synchronize()


def minibatch(proto_trees, Ns):
    """
    """
    
    global GWs, GW, GL, GV, Counters
    global GWs_gpu, GW_gpu, GL_gpu, GV_gpu
    global Ws, W, L, V
    global Ws_gpu, W_gpu, V_gpu

    
    THETA = np.r_[Ws.flatten(), W.flatten(), L.flatten(), V.flatten()]
    E = [0]*len(Ns)
    for k, proto_tree in enumerate(proto_trees):
        tree =  get_tree(deepcopy(proto_tree), Ns[k], vocabulary, L, params)
        forward(tree)
        backward(tree, Counters)
        T = np.vstack(tree.t)
        # TODO Remove this loop
        Y = np.empty((Ns[k], params['C']), dtype=params['dtype'])
        for row, y in enumerate(tree.y_gpu):
            Y[row, :] = y.get().astype(params['dtype'])
        E[k] = get_cost(T, Y, THETA, lmbd=params['lambda'])
                
    # Normalize the gradients
    GWs = GWs_gpu.get() / Counters['Ws']
    GW = GW_gpu.get() / Counters['W']
    GV = GV_gpu.get() / Counters['V']
    CL_bix = Counters['L'] > 0
    GL[CL_bix, :] = GL_gpu.get()[CL_bix, :] / \
        Counters['L'][CL_bix].reshape((CL_bix.sum(),1))
    # Update the weights
    Ws = Ws - params['eta'] * GWs - params['eta'] * params['lambda'] * Ws
    W = W - params['eta'] * GW - params['eta'] * params['lambda'] * W
    V = V - params['eta'] * GV - params['eta'] * params['lambda'] * V
    L = L - params['eta'] * GL - params['eta'] * params['lambda'] * L    

    GWs[:] = 0.0
    GW[:] = 0.0
    GV[:] = 0.0
    GL[:] = 0.0
    Counters['Ws'] = 0
    Counters['W'] = 0
    Counters['V'] = 0
    Counters['L'][:] = 0
    
    Ws_gpu = gpuarray.to_gpu(Ws)
    W_gpu = gpuarray.to_gpu(W)
    V_gpu = gpuarray.to_gpu(V)
    L_gpu = gpuarray.to_gpu(L)

    GWs_gpu = gpuarray.to_gpu(GWs)
    GW_gpu = gpuarray.to_gpu(GW)
    GV_gpu = gpuarray.to_gpu(GV)
    GL_gpu = gpuarray.to_gpu(GL)
    
    return E

    
def train():
    """
    For testing. Just one pass over the data set.
    """
    
    proto_trees, Ns = read_proto_trees(fn_train)
    n_trees = len(Ns)
    
    batch_Ids = np.arange(0, n_trees-params['BatchSz'], params['BatchSz'])
    np.random.shuffle(batch_Ids)
    for nbatch, idx0 in enumerate(batch_Ids):
        batch_trees = proto_trees[idx0:idx0+params['BatchSz']]
        ns = Ns[idx0:idx0+params['BatchSz']]
        E = minibatch(batch_trees, ns)
    
        print('Mini-batch # %d, average error: %0.6f' % (1+nbatch, np.mean(E)))
