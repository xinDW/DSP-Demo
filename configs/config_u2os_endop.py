
factor = 2
train_img_size_lr = [12, 32, 32, 1] #[d,h,w,c]
train_img_size_hr = [24, 64, 64, 1]



#label = 'U2OS_edoplasmetric_SRRF_8bit_2stage_dbpn+rdn_factor2_mse'   # final layer (of RDN) act-free, edoplasmetric
#label = 'U2OS_edoplasmetric_SRRF_8bit_2stage_dbpn+rdn_factor2_l1'
#label = 'U2OS_edoplasmetric_SRRF_8bit_1stage_rdn_factor2_mse'   # final layer (of RDN) act-free, 
label = 'U2OS_edoplasmetric_SRRF_8bit_fused_factor2_mse'

using_batch_norm = False
train_lr_img_path = "data/U2OS_edoplasmetric/8bit/LS/cropped32X32X12/"
train_hr_img_path = "data/U2OS_edoplasmetric/8bit/SRRF/cropped64X64X24/"
train_mr_img_path = "data/U2OS_edoplasmetric/8bit/SRRF/cropped64X64X24/ds/"


train_test_data_path = "data/U2OS_edoplasmetric/8bit/test/"
train_valid_lr_path = None #"data/bead_simu/valid_otf/"   # valid on_the_fly 


valid_lr_img_path = "data/U2OS_edoplasmetric/for_fig/g2/cropped50X50X53/"

valid_lr_img_size = [53,50,50,1] # [depth, height, width, channels]
