import os 
import tensorflow as tf

import numpy as np

import re
import time

from config import config
from utils import read_all_images, write3d, interpolate3d, get_file_list, load_im, _raise, exists_or_mkdir

from model.util import save_graph_as_pb, convert_graph_to_fp16, load_graph, Model, Predictor, LargeDataPredictor


using_batch_norm = config.using_batch_norm 

checkpoint_dir = config.TRAIN.ckpt_dir
pb_file_dir    = 'checkpoint/pb/'
lr_size        = config.VALID.lr_img_size 


device_id = config.TRAIN.device_id
conv_kernel = config.TRAIN.conv_kernel

label = config.label
archi1 = config.archi1
archi2 = config.archi2

factor = 1 if archi2 is 'unet' else config.factor


input_op_name     = 'Placeholder'
output_op_name  = 'net_s2/out/Tanh' # 

def build_model_and_load_npz(epoch, use_cpu=False, save_pb=False):
    from model import DBPN, res_dense_net, unet3d, denoise_net
    epoch = 'best' if epoch == 0 else epoch
    # # search for ckpt files 
    def _search_for_ckpt_npz(file_dir, tags):
        filelist = os.listdir(checkpoint_dir)
        for filename in filelist:
            if '.npz' in filename:
                if all(tag in filename for tag in tags):
                    return filename
        return None

    if (archi1 is not None):
        resolve_ckpt_file = _search_for_ckpt_npz(checkpoint_dir, ['resolve', str(epoch)])
        interp_ckpt_file  = _search_for_ckpt_npz(checkpoint_dir, ['interp', str(epoch)])
       
        (resolve_ckpt_file is not None and interp_ckpt_file is not None) or _raise(Exception('checkpoint file not found'))

    else:
        #checkpoint_dir = "checkpoint/" 
        #ckpt_file = "brain_conv3_epoch1000_rdn.npz"
        ckpt_file = _search_for_ckpt_npz(checkpoint_dir, [str(epoch)])
        
        ckpt_file is not None or _raise(Exception('checkpoint file not found'))
    

    #======================================
    # build the model
    #======================================
    
    if use_cpu is False:
        device_str = '/gpu:%d' % device_id
    else:
        device_str = '/cpu:0'

    LR = tf.placeholder(tf.float32, [1] + lr_size)
    if (archi1 is not None):
        # if ('resolve_first' in archi):        
        with tf.device(device_str):
            if archi1 is 'dbpn':   
                resolver = DBPN(LR, upscale=False, name="net_s1")
            elif archi1 is 'denoise': 
                resolver = denoise_net(LR, name="net_s1")
            else:
                _raise(ValueError())
            
            if archi2 is 'rdn':
                interpolator = res_dense_net(resolver.outputs, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm, is_train=False, name="net_s2")
                net = interpolator
            else:
                _raise(ValueError())

    else : 
        archi = archi2
        with tf.device(device_str):
            if archi is 'rdn':
                net = res_dense_net(LR, factor=factor, bn=using_batch_norm, conv_kernel=conv_kernel)
            elif archi is 'unet':
                net = unet3d(LR, upscale=False, is_train=False)
            elif archi is 'dbpn':
                net = DBPN(LR, upscale=True)
            else:
                raise Exception('unknow architecture: %s' % archi)

    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if (archi1 is None):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)

    return sess, net, LR

def save_as_pb(graph_file_tag, sess):
    tl.files.exists_or_mkdir(pb_file_dir)

    
    graph_file_16bit  = '%s_half-precision.pb' % (graph_file_tag)
    graph_file_32bit  = '%s.pb' % (graph_file_tag)
    graph_file = os.path.join(pb_file_dir, graph_file_32bit) 
    save_graph_as_pb(sess=sess, 
        output_node_names=output_op_name, 
        output_graph_file=graph_file)

    convert_graph_to_fp16(graph_file, pb_file_dir, graph_file_16bit, as_text=False, target_type='fp16', input_name=input_op_name, output_names=[output_op_name])


def evaluate_whole(model, half_precision_infer=False, use_cpu=False, large_volume=False, save_pb=True, save_activations=False):
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    start_time = time.time()
    
    device_tag = 'gpu' if not use_cpu else 'cpu'

    if model == 'brain':
        sample            = 'brain' 
        factor            = 4 
        lr_img_path = 'example_data/brain/LR/' 
        save_dir    = 'example_data/brain/SR/' 
    else:
        sample            = 'tubulin'
        factor            =  2
        lr_img_path = 'example_data/cell/LR/'
        save_dir    = 'example_data/cell/SR/' 

    graph_file_tag = '%s_2stage_dbpn+rdn_factor%d_50x50x50_%s' % (sample, factor, device_tag)
    graph_file     = '%s_half-precision.pb' % (graph_file_tag) if half_precision_infer else graph_file_tag + '.pb'
    

    model_path = os.path.join(pb_file_dir, graph_file)
    os.path.exists(model_path) or _raise(ValueError('%s doesn\'t exist' % model_path))

    import_name = "dsp"
    sess = load_graph(model_path, import_name=import_name, verbose=False)

    LR   = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, input_op_name))
    net  = sess.graph.get_tensor_by_name("%s/%s:0" % (import_name, output_op_name))

    exists_or_mkdir(save_dir)

    model      = Model(net, sess, LR)
    block_size = lr_size[0:3]
    overlap    = 0.2

    import imageio   
    dtype = np.float16 if half_precision_infer else np.float32
    if large_volume:
        start_time = time.time()
        predictor = LargeDataPredictor(data_path=lr_img_path, 
            saving_path=save_dir, 
            factor=factor, 
            model=model, 
            block_size=block_size,
            overlap=overlap,
            dtype=dtype)
        predictor.predict()
        print('time elapsed : %.2fs' % (time.time() - start_time))

    else:  
        valid_lr_imgs = get_file_list(path=lr_img_path, regx='.*.tif') 
        predictor = Predictor(factor=factor, model=model, dtype=dtype)

        for _, im_file in enumerate(valid_lr_imgs):
            start_time = time.time()
            im = imageio.volread(os.path.join(lr_img_path, im_file))
            
            print('='*66)
            print('predicting on %s ' % os.path.join(lr_img_path, im_file) )
            sr = predictor.predict(im, block_size, overlap, low=0.2)
            print('time elapsed : %.4f' % (time.time() - start_time))
            imageio.volwrite(os.path.join(save_dir, 'DSP_' + im_file), sr)
                
    model.recycle()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cpu", help="use cpu for inference",
                        action="store_true") 

    parser.add_argument("--brain", help="use the brain model for inference",
                        action="store_true") 

    parser.add_argument("--cell", help="use the cell model for inference", 
                        action="store_true")


    args = parser.parse_args()

    model = 'brain' if args.brain else 'cell'

    evaluate_whole(model=model, half_precision_infer=False, 
        use_cpu=args.cpu, 
        large_volume=False, 
        save_pb=False, 
        save_activations=False)
