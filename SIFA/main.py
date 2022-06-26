"""Code for training SIFA."""
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time
import nibabel as nib

###import tensorflow as tf
import tensorflow.compat.v1 as tf ###
tf.disable_v2_behavior() ###

import data_loader, losses, model
from stats_func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/targets/x86_64-linux/lib:' + os.environ['LD_LIBRARY_PATH']
#/usr/local/cuda/lib64:
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#os.system("ldd /usr/local/cuda/lib64/libcudnn.so.8 > rqndo.txt")

save_interval = 300 #300 ==> Save each 300 Epochs
evaluation_interval = 10 #10 ==> Evaluate loss each 10 epoch
random_seed = 1234
seg_param = 1 # Segmentation Losse ON/OFF
save_loss_cnt = 100 # Save losses each x iterations



class SIFA:
    """The SIFA module."""

   
    
    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_pth']
        
        self._target_train_pth = config['target_train_pth']
        self._source_val_pth = config['source_val_pth']
        self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)
        self._output_summary = os.path.join(self._output_dir, 'summary.txt') 
        self._nib_images_dir = os.path.join(self._output_dir, 'nib_imgs')
        if not os.path.exists(self._nib_images_dir):
                os.makedirs(self._nib_images_dir)
        self._num_imgs_to_save = 20 #_num_imgs_to_save
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._max_step = int(config['max_step']) #Number of "epochs"
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']
        print_summary(self._output_summary,"\nCALL SIFA CONSTRUCTOR ==> Take params from config file\n")

        ### Starts from images of zeros 
        
        self.fake_images_A = np.zeros((self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))
        
        self.fake_images_B = np.zeros((self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))
        
        

    
    
    
    
    
    
    
    def model_setup(self):
        #tf.compat.v1.placeholder()  <== tf.placeholder
        print_summary(self._output_summary,"\nModel Setup\n")
        
        self.input_a = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_A")
        
        
        first_dim = None 
        
        self.input_b = tf.compat.v1.placeholder(
            tf.float32, [
                None,  
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_B")
        
        self.fake_pool_A = tf.compat.v1.placeholder(
            tf.float32, [
                first_dim,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        
        self.fake_pool_B = tf.compat.v1.placeholder(
            tf.float32, [
                first_dim,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        
        self.gt_a = tf.compat.v1.placeholder(
            tf.float32, [
                first_dim,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        
        self.gt_b = tf.compat.v1.placeholder(
            tf.float32, [
                first_dim,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B") # for validation only, not used during training

        
        self.keep_rate = tf.compat.v1.placeholder(tf.float32, shape=())
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        self.num_fake_inputs = 0

        self.learning_rate_gan = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr_gan")
        self.learning_rate_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr_seg")

        self.lr_gan_summ = tf.summary.scalar("lr_gan", self.learning_rate_gan)
        self.lr_seg_summ = tf.summary.scalar("lr_seg", self.learning_rate_seg)

        inputs = {  #SELF.INPUTS 
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }
        

        outputs = model.get_outputs(inputs, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)
       
        
        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        
        self.pred_mask_a = outputs['pred_mask_a']
        self.pred_mask_b = outputs['pred_mask_b']
        self.pred_mask_b_ll = outputs['pred_mask_b_ll']
        self.pred_mask_fake_a = outputs['pred_mask_fake_a']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.pred_mask_fake_b_ll = outputs['pred_mask_fake_b_ll']
        
        self.prob_pred_mask_fake_b_is_real = outputs['prob_pred_mask_fake_b_is_real']
        self.prob_pred_mask_b_is_real = outputs['prob_pred_mask_b_is_real']
        self.prob_pred_mask_fake_b_ll_is_real = outputs['prob_pred_mask_fake_b_ll_is_real']
        self.prob_pred_mask_b_ll_is_real = outputs['prob_pred_mask_b_ll_is_real']

        self.prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
        self.prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
        self.prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']
        
        self.predicter_fake_b = tf.nn.softmax(self.pred_mask_fake_b)
        self.compact_pred_fake_b = tf.argmax(self.predicter_fake_b, 3)
        self.compact_y_fake_b = tf.argmax(self.gt_a, 3)

        self.predicter_b = tf.nn.softmax(self.pred_mask_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)

        self.dice_fake_b_arr = dice_eval(self.compact_pred_fake_b, self.gt_a, self._num_cls)
        self.dice_fake_b_mean = tf.reduce_mean(self.dice_fake_b_arr)
        self.dice_fake_b_mean_summ = tf.summary.scalar("dice_fake_b", self.dice_fake_b_mean)

        self.dice_b_arr = dice_eval(self.compact_pred_b, self.gt_b, self._num_cls)
        self.dice_b_mean = tf.reduce_mean(self.dice_b_arr)
        self.dice_b_mean_summ = tf.summary.scalar("dice_b", self.dice_b_mean)

    
    
    
    
    
    
    
    
    def compute_losses(self):  #==> CHANGE
        import tensorflow.compat.v1 as tf ###
        
        print_summary(self._output_summary,"\nCompute Losses\n")
        
        ############################# Cycle consistency Loss A and B ##################################
        cycle_consistency_loss_a = self._lambda_a * losses.cycle_consistency_loss(
                                                    real_images=self.input_a, generated_images=self.cycle_images_a)
        
        cycle_consistency_loss_b = self._lambda_b * losses.cycle_consistency_loss(
                                                     real_images=self.input_b, generated_images=self.cycle_images_b)
        
        
        ########################### GAN Loss ###########################################################
        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)
        

        ################ Segmentation Losses #######################################################
        lsgan_loss_p = losses.lsgan_loss_generator(self.prob_pred_mask_b_is_real) 
        lsgan_loss_p_ll = losses.lsgan_loss_generator(self.prob_pred_mask_b_ll_is_real)
        lsgan_loss_a_aux = losses.lsgan_loss_generator(self.prob_fake_a_aux_is_real)

        
        ce_loss_b, dice_loss_b = losses.task_loss(self.pred_mask_fake_b, self.gt_a)
        ce_loss_b_ll, dice_loss_b_ll = losses.task_loss(self.pred_mask_fake_b_ll, self.gt_a)
        
        l2_loss_b = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                              if '/s_B/' in v.name or '/s_B_ll/' in v.name or '/e_B/' in v.name])

        ##################### TOTAL GENERATOR LOSS #########################################################
        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a
        
        
        ################ TOTAL SEGMENTATION LOSS ####################################################
        seg_loss_B = seg_param*(ce_loss_b + dice_loss_b + l2_loss_b + ( \
                    + 0.1 * (ce_loss_b_ll + dice_loss_b_ll) + 0.1 * g_loss_B \
                    + 0.1 * lsgan_loss_p + 0.01 * lsgan_loss_p_ll + 0.1 * lsgan_loss_a_aux))

        ################ DISCRIMINATOR LOSS #############################################################à
        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,)
        
        d_loss_A_aux = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_cycle_a_aux_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_aux_is_real,)
        
        d_loss_A = d_loss_A + d_loss_A_aux
        
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,)
        
        d_loss_P = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_pred_mask_fake_b_is_real,
            prob_fake_is_real=self.prob_pred_mask_b_is_real,)
        
        d_loss_P_ll = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_pred_mask_fake_b_ll_is_real,
            prob_fake_is_real=self.prob_pred_mask_b_ll_is_real,)
        
        d_loss_P = seg_param * d_loss_P
        d_loss_P_ll = seg_param* d_loss_P_ll
        
        ################ Choosen Optimizer for GAN e SEG #####################################################
        optimizer_gan = tf.train.AdamOptimizer(self.learning_rate_gan, beta1=0.5)
        optimizer_seg = tf.train.AdamOptimizer(self.learning_rate_seg)

        
        
        self.model_vars = tf.trainable_variables() # ==> Trainable vars

        d_A_vars    = [var for var in self.model_vars if '/d_A/' in var.name]
        d_B_vars    = [var for var in self.model_vars if '/d_B/' in var.name]
        g_A_vars    = [var for var in self.model_vars if '/g_A/' in var.name]
        e_B_vars    = [var for var in self.model_vars if '/e_B/' in var.name]
        de_B_vars   = [var for var in self.model_vars if '/de_B/' in var.name]
        s_B_vars    = [var for var in self.model_vars if '/s_B/' in var.name]
        s_B_ll_vars = [var for var in self.model_vars if '/s_B_ll/' in var.name]
        d_P_vars    = [var for var in self.model_vars if '/d_P/' in var.name]
        d_P_ll_vars = [var for var in self.model_vars if '/d_P_ll/' in var.name]

        # Minimize Optimizer
        self.d_A_trainer    = optimizer_gan.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer    = optimizer_gan.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer    = optimizer_gan.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer    = optimizer_gan.minimize(g_loss_B, var_list=de_B_vars)
        self.d_P_trainer    = optimizer_gan.minimize(d_loss_P, var_list=d_P_vars)
        self.d_P_ll_trainer = optimizer_gan.minimize(d_loss_P_ll, var_list=d_P_ll_vars)
        
        self.s_B_trainer    = optimizer_seg.minimize(seg_loss_B, var_list=e_B_vars + s_B_vars + s_B_ll_vars)

        print_summary(self._output_summary,"\nMODEL VARs\n")
        for var in self.model_vars:
            print_summary(self._output_summary,f'{var.name}\n')

        # Summary variables
        if not os.path.exists(os.path.join(self._output_dir, 'losses')):
            os.makedirs(os.path.join(self._output_dir,'losses'))
            
        self.g_A_loss_summ = g_loss_A #tf.summary.scalar("g_A_loss", g_loss_A)
        self.lsgan_a = lsgan_loss_a
        self.consistency_loss_a = cycle_consistency_loss_a
        
        
        
        self.g_B_loss_summ = g_loss_B
        self.lsgan_b = lsgan_loss_b
        self.consistency_loss_b = cycle_consistency_loss_b
        
        self.d_A_loss_summ = d_loss_A
        self.d_B_loss_summ = d_loss_B
        
        
        self.ce_B_loss_summ = ce_loss_b
        self.dice_B_loss_summ = dice_loss_b
        
        self.l2_B_loss_summ = l2_loss_b
        self.s_B_loss_summ = seg_loss_B
        
        self.s_B_loss_merge_summ = [self.ce_B_loss_summ, self.dice_B_loss_summ, self.l2_B_loss_summ, self.s_B_loss_summ]
        self.d_P_loss_summ = d_loss_P

        self.d_P_ll_loss_summ = d_loss_P_ll
        self.d_P_loss_merge_summ = [self.d_P_loss_summ, self.d_P_ll_loss_summ] # tf.summary.merge([xxx,yyy])
        
        ############## Losses file ################################
        self._output_g_A_loss_summ = os.path.join(self._output_dir, 'losses/g_loss_A.txt')
        self._output_g_B_loss_summ = os.path.join(self._output_dir, 'losses/g_loss_B.txt')
        
        self._output_d_A_loss_summ = os.path.join(self._output_dir, 'losses/d_loss_A.txt')
        self._output_d_B_loss_summ  = os.path.join(self._output_dir, 'losses/d_loss_B.txt')
        
        self._output_ce_B_loss_summ = os.path.join(self._output_dir, 'losses/ce_loss_b.txt')
        self._output_dice_B_loss_summ = os.path.join(self._output_dir, 'losses/dice_loss_b.txt')
        
        self._output_l2_B_loss_summ = os.path.join(self._output_dir, 'losses/l2_loss_b.txt')
        self._output_s_B_loss_summ = os.path.join(self._output_dir, 'losses/seg_loss_B.txt')
        
        self._output_s_B_loss_merge_summ = os.path.join(self._output_dir, 'losses/s_B_loss_merge_summ.txt')
        self._output_d_P_loss_summ = os.path.join(self._output_dir, 'losses/d_loss_P.txt')
        
        self._output_d_P_ll_loss_summ = os.path.join(self._output_dir, 'losses/d_loss_P_ll.txt')
        self._output_d_P_loss_merge_summ = os.path.join(self._output_dir, 'losses/d_P_loss_merge_summ.txt')
        
        print_summary(self._output_s_B_loss_merge_summ,"[  ce_B  ,  dice_B ,   l2_B  ,  seg_B  ]",False)
    
    
    
    
    
    def save_images(self, sess, step):

        names = ['inputA_', 'inputB_', 'fakeA_','fakeB_', 'cycA_', 'cycB_']
        print_summary(self._output_summary,f'\nSave Images\n')

        with open(os.path.join(self._output_dir, 'step_' + str(step) + '.html'), 'w') as v_html: #./output/current_time/Step...
            for i in range(0, self._num_imgs_to_save): # Defined in the constructor

                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
         
                inputs = {
                    'images_i': images_i,
                    'images_j': images_j,
                    'gts_i': gts_i,
                    'gts_j': gts_j,
                }

                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j'],
                    self.is_training:self._is_training_value,
                    self.keep_rate: self._keep_rate_value,
                })

                # List of numpy array
                tensors = [inputs['images_i'], inputs['images_j'],fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]
                #N.B. Inconsistency beetween tensors and names ==> Follow the below notation
                
                #s:source
                #t:target
                #s>t transformed image from source to target (and viceversa t>s)
                #s~ cycle_transformed image (s>t>s) 
                #t~ cycle_transformed image (t>s>t)
                
                #names =  [     'inputA_'    ,     'inputB_'     ,  'fakeA_' ,  'fakeB_'  ,  'cycA_'  ,  'cycB_'  ]
                
                #         [       'X_s'      ,        'X_t'      ,  'X_s>t'   ,  'X_t>s'  ,  'X_s~'   ,  'X_t~'   ]
                
                
               
                

                for name, tensor in zip(names, tensors):
                    
                    #Save in jpg, html page
                    image_name = name + str(step) + "_" + str(i) + ".jpg"
                    #Conversion from [-1;1] to [0;255]
                    cv2.imwrite(os.path.join(self._images_dir, image_name),((tensor[0] + 1) *127.5).astype(np.uint8).squeeze()) 
                    
                    v_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                    
                    #Save in .nii.gz
                    nib_image_name = name + str(step) + "_" + str(i) + ".nii.gz"
                    nib.save(nib.Nifti1Image(tensor+1,np.eye(tensor.ndim)), os.path.join(self._nib_images_dir, nib_image_name))
                    #tensor.shape ==> (batch size = 4, 352, 352, 1)  
                    
                    #Future approach: try to reconstruct the entire volume:
                    #Load sorted(!) batch in order to save images from the same patient
                    
                v_html.write("<br>")

                
                
                
                ##### Save segmentation Mask #####
                # See model.py build_encoder() returns for further details
                '''
                    The encoder produce 2 different latent outputs:
                    o_r12 ==> before  DRN layer (Dilated Residual Networks) ==> the one indicate with the name "ll"
                    o_c3 ==> After 2 DRN + 2 Conv ==> Final output of the encoder

                '''
                
                # See model.py get_outputs() returns for further details
                ''' 
                   'pred_mask_a' = 'pred_mask_b': Predicted mask for the X_t image (Pred Y_t)
                   'pred_mask_b_ll': Predicted mask for the X_s_ll // X_t>s image (Pred Y_t>s)
                   'pred_mask_fake_a' = 'pred_mask_fake_b': Predicted mask for the X_s image (Pred Y_s)
                   'pred_mask_fake_b_ll': Predicted mask for the X_s>t_ll // X_s>t image (Pred Y_s>t)
                   'gt':Ground Truth, Real segmentation mask of the X_s image (Real Y_s)
                '''
                mask_names =['pred_mask_a', 'pred_mask_b', 'pred_mask_b_ll', 'pred_mask_fake_a' ,'pred_mask_fake_b',
                            'pred_mask_fake_b_ll','gt']
                pred_mask_a_temp, pred_mask_b_temp , pred_mask_b_ll_temp, \
                pred_mask_fake_a_temp, pred_mask_fake_b_temp, pred_mask_fake_b_ll_temp = sess.run([
                    self.pred_mask_a, self.pred_mask_b ,
                    self.pred_mask_b_ll,self.pred_mask_fake_a,
                    self.pred_mask_fake_b,self.pred_mask_fake_b_ll],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.gt_a:inputs['gts_i'],
                        #self.learning_rate_gan: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    })
            
                mask_tensors = [pred_mask_a_temp, pred_mask_b_temp , pred_mask_b_ll_temp,
                           pred_mask_fake_a_temp, pred_mask_fake_b_temp, pred_mask_fake_b_ll_temp,inputs['gts_i']]
            
                for name, tensor in zip(mask_names, mask_tensors):
                    
                    #Save in jpg, html page
                    image_name = name + str(step) + "_" + str(i) + ".jpg"
                    #Conversion from [0;1] to [0;255]
                    tensor =np.argmax(tensor,axis=-1,keepdims=True)
                    cv2.imwrite(os.path.join(self._images_dir, image_name),(tensor[0]*255).astype(np.uint8).squeeze()) 
                    
                    v_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                    
                    #Save in .nii.gz
                    nib_image_name = name + str(step) + "_" + str(i) + ".nii.gz"
                    nib.save(nib.Nifti1Image(tensor+1,np.eye(tensor.ndim)), os.path.join(self._nib_images_dir, nib_image_name))
                v_html.write("<br>")
                v_html.write("<br>")
               
                
        
    
    
    
    
    
    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    
    
    
    
    
    
    
    def train(self):
        import tensorflow.compat.v1 as tf ###
        # Load Dataset
        print_summary(self._output_summary,"\nCall train ==> Loading Dataset....\n")
        self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
        self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)
        
        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        print_summary(self._output_summary,"\nGLOBAL VARs Initialization\n")
        
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=40)

        with open(self._source_train_pth, 'r') as fp:
            rows_s = fp.readlines()
        with open(self._target_train_pth, 'r') as fp:
            rows_t = fp.readlines()
        
        ############### Open GPU Session ##############################
        gpu_options = tf.GPUOptions(allow_growth=True)
        print_summary(self._output_summary,"\nOpening GPU session...\n")
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            print_summary(self._output_summary,"\nGPU Opened !\n")
        
            cnt = -1
            ########### Restore the model to run the model from last checkpoint ##################
            if self._to_restore:
                print_summary(self._output_summary,"\nRestore model from last checkpoint\n")
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                cnt=int(chkpt_fname.split("sifa-")[1])
                saver.restore(sess, chkpt_fname)
            
            ############ Create output dir #######################################################
            writer = tf.summary.FileWriter(self._output_dir)
            writer_val = tf.summary.FileWriter(self._output_dir+'/val')

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            # Training Loop
            
            print_summary(self._output_summary,"\nSTART TRAINING\n")
            
            curr_lr_seg = 0.001

            BEST={"A":0,"B":0}
            for i in range(cnt+1,self._max_step):
                starttime = time.time()

                cnt += 1
                curr_lr = self._base_lr
                
                
                
                #Training Images
                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                inputs = {
                    'images_i': images_i,
                    'images_j': images_j,
                    'gts_i': gts_i,
                    'gts_j': np.zeros(gts_j.shape), 
                    # Zero shape mask for the target groundh truth (Already done in the preprocessing)
                }
                
                #Validation Images
                images_i_val, images_j_val, gts_i_val, gts_j_val = sess.run(self.inputs_val)
                inputs_val = {
                    'images_i_val': images_i_val,
                    'images_j_val': images_j_val,
                    'gts_i_val': gts_i_val,
                    'gts_j_val': gts_j_val,
                }
                
                # Inputs ==> Batch (!) of images + labels (n= 4 or 8,352,352,...)  
                
                # 1)Optimizing the G_A network  ==> Gt 
                _, fake_B_temp, summary_str, summary_lsgan_a, summary_consistency_a = sess.run(
                    [self.g_A_trainer,self.fake_images_b, self.g_A_loss_summ,self.lsgan_a,self.consistency_loss_a],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.gt_a:inputs['gts_i'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate:self._keep_rate_value,
                        self.is_training:self._is_training_value,
                    }
                )
                
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_g_A_loss_summ,
                                  f"[{summary_lsgan_a:.4f},{summary_consistency_a:.4f},{summary_str:.4f}], iter: {cnt}",False) 

                fake_B_temp1 = self.fake_image_pool(
                    self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                # 2)Optimizing the D_B network ==> Dt
                _, summary_str = sess.run(
                    [self.d_B_trainer, self.d_B_loss_summ],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.fake_pool_B: fake_B_temp1,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_d_B_loss_summ,f"{summary_str:.4f}, iter: {cnt}",False) 

                # 3)Optimizing the S_B network ==> Segmentation C
                _, summary_str = sess.run(
                    [self.s_B_trainer, self.s_B_loss_merge_summ],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.gt_a:inputs['gts_i'],
                        self.learning_rate_seg: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }

                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    summary_str = [format(x, '.4f') for x in summary_str]
                    print_summary(self._output_s_B_loss_merge_summ,f"{summary_str}, iter: {cnt}",False) 
                
                # 4)Optimizing the G_B network ==> E+U
                _, fake_A_temp, summary_str,summary_lsgan_b, summary_consistency_b = sess.run(
                    [self.g_B_trainer,self.fake_images_a,self.g_B_loss_summ,self.lsgan_b,self.consistency_loss_b],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.gt_a: inputs['gts_i'],
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_g_B_loss_summ,f"[{summary_lsgan_b:.4f},{summary_consistency_b:.4f},{summary_str:.4f}], iter: {cnt}",False) 
               
                

                fake_A_temp1 = self.fake_image_pool(
                    self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                # 5)Optimizing the D_A network ==> Ds
                _, summary_str = sess.run(
                    [self.d_A_trainer, self.d_A_loss_summ],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.fake_pool_A: fake_A_temp1,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_d_A_loss_summ,f"{summary_str:.4f}, iter: {cnt}",False) 

                # 6)Optimizing the D_P network ==> Dp
                _, summary_str = sess.run(
                    [self.d_P_trainer, self.d_P_loss_summ],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_d_P_loss_summ,f"{summary_str:.4f}, iter: {cnt}",False) 

                # 7)Optimizing the D_P_ll network
                _, summary_str = sess.run(
                    [self.d_P_ll_trainer, self.d_P_ll_loss_summ],
                    feed_dict={
                        self.input_a:inputs['images_i'],
                        self.input_b:inputs['images_j'],
                        self.learning_rate_gan: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                #writer.add_summary(summary_str, cnt)
                if cnt % save_loss_cnt == 0:
                    print_summary(self._output_d_P_ll_loss_summ,f"{summary_str:.4f}, iter: {cnt}",False) 

                summary_str_gan, summary_str_seg = sess.run(
                    [self.lr_gan_summ, self.lr_seg_summ],
                         feed_dict={
                             self.learning_rate_gan: curr_lr,
                             self.learning_rate_seg: curr_lr_seg,
                         })

                #writer.add_summary(summary_str_gan, cnt)
                #writer.add_summary(summary_str_seg, cnt)

                #writer.flush()
                self.num_fake_inputs += 1

                print_summary(self._output_summary,'iter {}: processing time {}'.format(cnt, time.time() - starttime))
                
                # batch evaluation
                if (i + 1) % evaluation_interval == 0: # Each 10 
                    print_summary(self._output_summary,"\nBatch Evaluation:")
                    
                    summary_str_fake_b, summary_str_b = sess.run([self.dice_fake_b_mean_summ, self.dice_b_mean_summ],
                                                                 feed_dict={
                                                                     self.input_a: inputs['images_i'],
                                                                     self.gt_a: inputs['gts_i'],
                                                                     self.input_b: inputs['images_j'],
                                                                     self.gt_b: inputs['gts_j'],
                                                                     self.is_training: False,
                                                                     self.keep_rate: 1.0,
                                                                 })
                    #writer.add_summary(summary_str_fake_b, cnt) 
                    #writer.add_summary(summary_str_b, cnt)
                    #writer.flush()

                    print_summary(self._output_summary,"\nDice Loss ==> {A: dice_fake_b, B: dice_b)\n")
                    
                    dice_fake_b, summary_str_fake_b, dice_b, summary_str_b = sess.run(
                            [self.dice_fake_b_mean, self.dice_fake_b_mean_summ, self.dice_b_mean, self.dice_b_mean_summ],                           feed_dict={
                            self.input_a: inputs_val['images_i_val'],
                            self.gt_a: inputs_val['gts_i_val'],
                            self.input_b: inputs_val['images_j_val'],
                            self.gt_b: inputs_val['gts_j_val'],
                            self.is_training: False,
                            self.keep_rate: 1.0,
                        })
                    #writer_val.add_summary(summary_str_fake_b, cnt)
                    #writer_val.add_summary(summary_str_b, cnt)
                    #writer_val.flush()

                    #dice_fake_b ==> segmentation A
                    #dice_b ==> segmentation B
                    
                    if(BEST["A"]<=dice_fake_b):
                        print_summary(self._output_summary,"NEW BEST_A AT ITER {}: {}".format(cnt,dice_fake_b))
                        BEST["A"]=dice_fake_b
                        self.save_images(sess, "A")
                        saver.save(sess, os.path.join(self._output_dir, "sifa_A"), global_step=None)
                    
                    if(BEST["B"]<=dice_b):
                        print_summary(self._output_summary,"NEW BEST_B AT ITER {}: {}".format(cnt,dice_b))
                        BEST["B"]=dice_b
                        self.save_images(sess, "B")
                        saver.save(sess, os.path.join(self._output_dir, "sifa_B"), global_step=None)

                if (cnt+1) % save_interval ==0: #Save each 300
                    print_summary(self._output_summary,"\nSave Images\n")
                    self.save_images(sess, cnt)
                    saver.save(sess, os.path.join(self._output_dir, "sifa"), global_step=cnt)

            coord.request_stop()
            coord.join(threads)
            #writer.add_graph(sess.graph)



            
            
# Print in the console and also in a file
def print_summary(filename, line,print_here=True):
    if print_here:
        print(line)
    with open(filename, 'a+') as f:
        f.write(str(line))   
        f.write("\n")
            


    
    
def main(config_filename):
    
    #tf.set_random_seed(random_seed) ==> Deprecated
    tf.compat.v1.set_random_seed(random_seed)

    with open(config_filename) as config_file:
        config = json.load(config_file)
    
    sifa_model = SIFA(config)
    sifa_model.train()    

if __name__ == '__main__':
    main(config_filename='./config_param.json')
