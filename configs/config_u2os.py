
factor = 2
train_img_size_lr = [32, 32, 32, 1] #[d,h,w,c]
train_img_size_hr = [64, 64, 64, 1]

label = 'U2OS_tubulin_SRRF_low_snr_input_8bit'#_2stage_dbpn+rdn_factor2_mse' # best fine-tuned-110   # final layer (of RDN) act-free 


archi2 = 'rdn'  #['rdn', 'unet', 'dbpn']
archi1 = 'dbpn'   # [None, 'dbpn' 'denoise'] # None if 1stage
# archi1 = 'denoise'

loss      = 'mse'  #['mse', 'mae']
archi_str = '2stage_{}+{}'.format(archi1, archi2) if archi1 is not None else '1stage_{}'.format(archi2)
label     = '{}_{}_factor{}_{}'.format(label, archi_str, factor, loss)

using_batch_norm = False

train_lr_img_path = "data/U2OS_tubulin/20200118/LR_cropped32X32X32/"
train_hr_img_path = "data/U2OS_tubulin/20200118/HR_cropped64X64X64/"
train_mr_img_path = "data/U2OS_tubulin/20200118/MR_cropped32X32X32/"




train_test_data_path = None
train_valid_lr_path = None #"data/bead_simu/valid_otf/"   # valid on_the_fly 



#config.VALID.lr_img_path = "data/celegans/A+B/period2/period2_8bit_roi/cropped100X100X13/"

#config.VALID.lr_img_path = "data/U2OS_tubulin/for_fig3/high_snr/"
#config.VALID.lr_img_path = "data/U2OS_tubulin/SIM/group3/cropped100X100X18/"
#config.VALID.lr_img_path = "data/U2OS_edoplasmetric/for_fig/g1/cropped50X50X46/"

valid_lr_img_path = "example-data/test/cell/LR/"
# valid_lr_img_path = "data/SI-transfer-boundary/data/LR/"

# valid_lr_img_path = "J:/data_se_20191124_/lr20191205/"
valid_lr_img_size = [50,50,50,1] # [depth, height, width, channels]
