
factor = 4
train_img_size_lr = [20, 20, 20, 1] #[d,h,w,c]
train_img_size_hr = [80, 80, 80, 1]



label = 'whole+half_brain_training_hr_shallow_region_8bit_step1um' #_2stage_dbpn+rdn_factor4_mse'

archi1 = 'dbpn'
archi2 = 'rdn'

loss      = 'mse'  #['mse', 'mae']
archi_str = '2stage_{}+{}'.format(archi1, archi2) if archi1 is not None else '1stage_{}'.format(archi2)
label     = '{}_{}_factor{}_{}'.format(label, archi_str, factor, loss)

using_batch_norm = False

train_lr_img_path = "data/brain/brain20190316/8bit/cropped80X80X80/lr/all/"
train_hr_img_path = "data/brain/brain20190316/8bit/cropped80X80X80/hr/all/"
train_mr_img_path = None

train_test_data_path = None
train_valid_lr_path = None  # valid on_the_fly 

valid_lr_img_path = "example-data/brain/test/brain/LR"
# valid_lr_img_path = "F:/DVSR/thy1_3.2x/squence/"
valid_lr_img_size = [25,50,50,1]


