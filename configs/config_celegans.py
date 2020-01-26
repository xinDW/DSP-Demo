
factor = 4
train_img_size_lr = [10, 24, 24, 1] #[d,h,w,c]
train_img_size_hr = [40, 96, 96, 1]

using_batch_norm = False
#label = 'celegans_simu_by_zhaoyuxuan_2stage_dbpn+rdn_factor4_mse'
#label = 'bead_simu_8bit_1stage_rdn_factor4_mse'

label = 'bead_simu_8bit_2stage_dbpn+rdn_factor4_mse'    # final layer (of RDN) act-free
label = 'bead_simu_8bit_2stage_dbpn+rdn_factor4_mse/tuned-with-oblique'

#label = 'bead_simu_8bit_2stage_dbpn+rdn_factor4_mse/z_extend_mid'
#label = 'celegans_panneu_16X_confocal_step1um_2stage_dbpn+rdn_factor4_mse'       # final layer (of RDN) act-free
#label = 'celegans_panneu_16x_step1um_1stage_rdn_factor4_mse'       # final layer (of RDN) act-free


archi2 = 'rdn'  #['rdn', 'unet', 'dbpn']
# archi1 = 'dbpn'   # [None, 'dbpn' 'denoise'] # None if 1stage
archi1 = 'dbpn'

loss      = 'mse'  #['mse', 'mae']
archi_str = '2stage_{}+{}'.format(archi1, archi2) if archi1 is not None else '1stage_{}'.format(archi2)
# label     = '{}_{}_factor{}_{}'.format(label, archi_str, factor, loss)



train_lr_img_path = "data/bead_simu/oblique/LR/"
train_hr_img_path = "data/bead_simu/oblique/HR/"
train_mr_img_path = "data/bead_simu/oblique/MR/"

train_test_data_path = None
train_valid_lr_path = "data/bead_simu/valid_otf/"  # valid on_the_fly 

valid_lr_img_path = "data/celegans/A+B/period3/period3_16bit_stack/test/"
valid_lr_img_path = "data/celegans/A+B/period2/8bit_stack/"
#valid_lr_img_path = "data/celegans/A+B/period1/period1-1/8bit-roi/cropped100X100X13/"
valid_lr_img_size = [13,100,100,1] # [depth, height, width, channels]
