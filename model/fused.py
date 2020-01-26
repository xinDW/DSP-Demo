import tensorflow as tf
import numpy as np

from .custom import conv3d
from .dbpn import DBPN_front
from .res_dense_net import RDN_end

def fusedSegNet(input, factor=4, reuse=False, name='fused'):
    removedLayerDBPN = [i for i in range(54, 76)]
    removedLayerRDN = [i for i in range(0, 4)] + [32, 33]
    removed = {'dbpn': removedLayerDBPN, 'rdn' : removedLayerRDN}
    with tf.variable_scope(name, reuse=reuse):
        resovlerFront = DBPN_front(input, reuse=reuse, name='dbpn')
        connector     = conv3d(resovlerFront, out_channels=64, filter_size=3, name='connector')
        interperEnd   = RDN_end(connector, factor=factor, reuse=reuse, name='rdn')
    return interperEnd, resovlerFront, removed


def load_ckpt_partial(ckpt_path, net, begin, removed, sess):
    """
    load an npz ckpt file and assign to the 'net'
    Params:
        -removed: layers of which the weights and bias are removed from net, but still saved in npz file
    """
    
    d = np.load( ckpt_path, encoding='latin1')
    params = d['params']    

    ops = []
    i = begin
    for idx, param in enumerate(params):
        if idx not in removed:
            print('loading %d : %s' % (idx, str(param.shape)))
            ops.append(net.all_params[i].assign(param)) 
            i += 1
        else:
            print('omitting %d : %s' % (idx, str(param.shape)))
    if sess is not None:
        sess.run(ops)
   

