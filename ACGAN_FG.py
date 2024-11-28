import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, ReLU,LayerNormalization,BatchNormalization,Conv1D,Reshape,concatenate,Flatten, Dropout, Concatenate,multiply
from tensorflow.keras.models import Sequential, Model
import datetime
import read_data
from tensorflow.keras.losses import mean_squared_error
from new_zero_shot_13_1_evaluation import feature_generation_and_diagnosis


class RandomWeightedAverage(Concatenate):
    """Provides a (random) weighted average between real and generated samples"""
    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        alpha = K.random_uniform((batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
class Zero_shot():
    def __init__(self):
        self.data_lenth=52
        self.sample_shape=(self.data_lenth,)
        
        self.feature_dim=256
        self.feature_shape=(256,)
        self.num_classes=15
        self.latent_dim = 50
        self.noise_shape=(self.latent_dim,1)
        self.n_critic = 1  #randomly select from (1,5) 
        self.LAMBDA_GP=10  #randomly select from (5,15) 
        self.num_blocks=3  #randomly select from (1,6) 
        self.crl = False   #self.crl =True

        self.lambda_cla = 10 #randomly select from (1,15)
        self.lambda_cms = 10 #randomly select from (1,15)
        self.lambda_crl = 0.01 #randomly select from (0,3)
        
        self.bound = False #self.bound =True
        self.mi_weight = 0.001 #randomly select from (0,1)
        self.mi_bound = 100 #randomly select from (10,100)
        
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # range from (1e-2,1e-4)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # range from (1e-2,1e-4)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # range from (1e-2,1e-4)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)# range from (1e-2,1e-4)
        self.m_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)# range from (1e-2,1e-4)
        
        self.autoencoder= self.build_autoencoder()
        self.d = self.build_discriminator()
        self.g = self.build_generator()
        self.c= self.build_classifier()
        self.m= self.build_comparator()
 

        self.autoencoder.trainable = False
        self.c.trainable = False
        self.m.trainable = True
        self.d.trainable = False
        self.g.trainable = False

        sample_1= Input(shape=self.sample_shape)
        feature_1, output_sample_1=self.autoencoder(sample_1)

        sample_2= Input(shape=self.sample_shape)
        feature_2, output_sample_1=self.autoencoder(sample_2)

        predicted_similarity=self.m([feature_1,feature_2])

        self.m_model=Model(inputs=[sample_1,sample_2], outputs=[predicted_similarity])
        self.m_model.compile(loss=['binary_crossentropy'], optimizer=self.m_optimizer)
             

    def build_autoencoder(self):
      
      
      sample = Input(shape=self.sample_shape)     
     
      a0=sample

      # Encoder

      a1=Dense(100)(a0)
      a1=LeakyReLU(alpha=0.2)(a1)
      a1=LayerNormalization()(a1)

      a2=Dense(200)(a1)
      a2=LeakyReLU(alpha=0.2)(a2)
      a2=LayerNormalization()(a2)

      a3=Dense(256)(a2)
      a3=LeakyReLU(alpha=0.2)(a3)
      a3=LayerNormalization()(a3)

      feature=a3

      # Decoder

      a4=Dense(200)(feature)
      a4=LeakyReLU(alpha=0.2)(a4)
      a4=LayerNormalization()(a4)

      a5=Dense(100)(a4)
      a5=LeakyReLU(alpha=0.2)(a5)
      a5=LayerNormalization()(a5)

      a6=Dense(52)(a5)
      a6=LeakyReLU(alpha=0.2)(a6)
      a6=LayerNormalization()(a6)

      output_sample=a6


    
      # Autoencoder Model
      autoencoder = Model(sample,[feature, output_sample])
      return autoencoder    


        
    def build_discriminator(self):
        

        sample_input = Input(shape=self.feature_shape)
        reshaped_sample = Reshape((256, 1))(sample_input)

        attribute = Input(shape=(20,), dtype='float32')
        reshape_attribute= Reshape((20,1))(attribute)

        concatenated = concatenate([reshaped_sample, reshape_attribute], axis=1)

        d0=Flatten()(concatenated)

        d1=Dense(200)(d0)
        d1=LeakyReLU(alpha=0.2)(d1)
        d1=LayerNormalization()(d1)

        d2=Dense(100)(d1)
        d2=LeakyReLU(alpha=0.2)(d2)
        d2=LayerNormalization()(d2)

               
        validity = Dense(1)(d2)

        return Model([sample_input,attribute],validity)

    def build_generator(self):


      
      noise = Input(shape=self.noise_shape)
      
      attribute = Input(shape=(20,), dtype='float32')
      reshape_attribute= Reshape((20,1))(attribute)

      concatenated = concatenate([noise, reshape_attribute], axis=1)

      g0=Flatten()(concatenated)

      g1=Dense(100)(g0)
      g1=LeakyReLU(alpha=0.2)(g1)
      g1=LayerNormalization()(g1)

      g2=Dense(200)(g1)
      g2=LeakyReLU(alpha=0.2)(g2)
      g2=LayerNormalization()(g2)

      g3=Dense(256)(g2)
      g3=LeakyReLU(alpha=0.2)(g3)
      
      
      generated_feature=g3

      generated_feature=BatchNormalization()(generated_feature)

      return Model([noise,attribute],generated_feature)


    
    def build_classifier(self):
        
        
        sample = Input(shape=self.feature_shape)

        c0=sample

        c1=Dense(100)(c0)
        c1=LeakyReLU(alpha=0.2)(c1)

        
        c2=Dense(50)(c1)
        c2=LeakyReLU(alpha=0.2)(c2)

        hidden_ouput=c2
               
               
        c3 = Dense(20,activation="sigmoid")(c2)
        

        predict_attribute=c3
        
        return Model(sample,[hidden_ouput,predict_attribute])

    def build_comparator(self):

      def conv_block(x, filters, stride, kernel_size=3):
        
        x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

      sample_1 = Input(shape=self.feature_shape)

      sample_2 = Input(shape=self.feature_shape)

      s1=Reshape((256,1))(sample_1)

      s2=Reshape((256,1))(sample_2)

      concatenated=concatenate([s1, s2], axis=-1)

      c0=concatenated

      c1=conv_block(c0,filters=16,stride=2)

      c2=conv_block(c1,filters=32,stride=2)

      c3=conv_block(c2,filters=64,stride=1)

      c4=Flatten()(c3)

      c5=Dense(1500)(c4)
      c5=LeakyReLU(alpha=0.2)(c5)
      c5=LayerNormalization()(c5)

      c6=Dense(100)(c5)
      c6=LeakyReLU(alpha=0.2)(c6)
      c6=LayerNormalization()(c6)

      c7=Dense(1,activation="sigmoid")(c6)

      similarity=c7

      return Model([sample_1,sample_2],similarity)



    
    
    def gradient_penalty_loss(self, gradients):
      
      gradients_sqr = tf.square(gradients)
      gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, len(gradients_sqr.shape)))
      gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
      gradient_penalty = tf.square(1 - gradient_l2_norm)
      return tf.reduce_mean(gradient_penalty)
    
    def wasserstein_loss(self, y_true, y_pred):
            return K.mean(y_true * y_pred)
          
    def estimate_mutual_information(self, x, z):
      # Create shuffled version of z to get independent samples
        z_shuffle = tf.random.shuffle(z)
        
        joint = tf.concat([x, z], axis=-1)
        marginal = tf.concat([x, z_shuffle], axis=-1)
        
        # Statistics network for MI estimation
        def statistics_network(samples):
            h1 = Dense(64, activation='relu')(samples)
            h2 = Dense(32, activation='relu')(h1)
            return Dense(1)(h2)
        
        t_joint = statistics_network(joint)
        t_marginal = statistics_network(marginal)
        
        mi_est = tf.reduce_mean(t_joint) - tf.math.log(tf.reduce_mean(tf.exp(t_marginal)))
        return mi_est
    
    def mi_penalty_loss(self, x, z):
        
        mi_est = self.estimate_mutual_information(x, z)
        return tf.maximum(0.0, mi_est - self.mi_bound)  
    
    def classification_loss(self,current_batch_features,y_true, hidden_output, pred_attribute):
                
        # Original classification loss (binary cross-entropy)
        classification_loss = tf.keras.losses.binary_crossentropy(
                y_true, pred_attribute)
        
        mi_penalty=0    
        # MI penalty
        if self.bound == True:    
        # MI penalty
          mi_penalty = self.mi_penalty_loss(
              current_batch_features, hidden_output)
            
         # Combined loss
        total_loss = classification_loss + self.mi_weight * mi_penalty
            
        return total_loss
    
    def comparison_loss(self,feature,similar_feature,unsimilar_feature,similar_truth,unsimilar_truth):
      
      predicted_similarity_simi=self.m([feature,similar_feature])
      predicted_similarity_unsimi=self.m([feature,unsimilar_feature])
      
      comparison_loss_simi=tf.keras.losses.binary_crossentropy(
                similar_truth, predicted_similarity_simi)
      comparison_loss_unsimi=tf.keras.losses.binary_crossentropy(
                unsimilar_truth, predicted_similarity_unsimi)        
      
      total_loss = comparison_loss_simi + comparison_loss_unsimi
      
      return total_loss
    
    def cycle_rank_loss(self,generated_feature,reconstructed_feature,unsimilar_generated_feature,similar_truth):
      
      predicted_similarity_simi=self.m([generated_feature,reconstructed_feature])
      predicted_similarity_unsimi=self.m([unsimilar_generated_feature,reconstructed_feature])
      
      comparison_loss_simi=tf.keras.losses.binary_crossentropy(
                similar_truth, predicted_similarity_simi)
      comparison_loss_unsimi=tf.keras.losses.binary_crossentropy(
                similar_truth, predicted_similarity_unsimi)
      
      loss = tf.maximum(0, comparison_loss_simi - comparison_loss_unsimi)
      
      return loss    
    
    def train(self, epochs, batch_size):
        
        start_time = datetime.datetime.now()
        
        accuracy_list_1=[]
        accuracy_list_2=[]
        accuracy_list_3=[]
        accuracy_list_4=[]
        
        # Adversarial loss ground truths
        valid = -np.ones((batch_size,1) )
        fake = np.ones((batch_size,1) )
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        similar_truth=np.ones((batch_size,1) )
        unsimilar_truth=np.zeros((batch_size,1) )
        
        PATH_train='./dataset_train_case1.npz'
        PATH_test='./dataset_test_case1.npz'
        
        train_data = np.load(PATH_train)
        test_data = np.load(PATH_test)
        
        train_sample_1=train_data['training_samples_1']
        train_attribute_1=train_data['training_attribute_1']

        train_sample_2=train_data['training_samples_2']
        train_attribute_2=train_data['training_attribute_2']

        train_sample_3=train_data['training_samples_3']
        train_attribute_3=train_data['training_attribute_3']

        train_sample_4=train_data['training_samples_4']
        train_attribute_4=train_data['training_attribute_4']

        train_sample_5=train_data['training_samples_5']
        train_attribute_5=train_data['training_attribute_5']

        train_sample_6=train_data['training_samples_6']
        train_attribute_6=train_data['training_attribute_6']

        train_sample_7=train_data['training_samples_7']
        train_attribute_7=train_data['training_attribute_7']

        train_sample_8=train_data['training_samples_8']
        train_attribute_8=train_data['training_attribute_8']

        train_sample_9=train_data['training_samples_9']
        train_attribute_9=train_data['training_attribute_9']

        train_sample_10=train_data['training_samples_10']
        train_attribute_10=train_data['training_attribute_10']

        train_sample_11=train_data['training_samples_11']
        train_attribute_11=train_data['training_attribute_11']

        train_sample_12=train_data['training_samples_12']
        train_attribute_12=train_data['training_attribute_12']

        train_sample_13=train_data['training_samples_13']
        train_attribute_13=train_data['training_attribute_13']

        train_sample_14=train_data['training_samples_14']
        train_attribute_14=train_data['training_attribute_14']

        train_sample_15=train_data['training_samples_15']
        train_attribute_15=train_data['training_attribute_15']


        test_sample_1=test_data['testing_samples_1']
        test_attribute_1=test_data['testing_attribute_1']

        test_sample_2=test_data['testing_samples_2']
        test_attribute_2=test_data['testing_attribute_2']
                
        test_sample_3=test_data['testing_samples_3']
        test_attribute_3=test_data['testing_attribute_3']

        test_sample_4=test_data['testing_samples_4']
        test_attribute_4=test_data['testing_attribute_4']

        test_sample_5=test_data['testing_samples_5']
        test_attribute_5=test_data['testing_attribute_5']

        test_sample_6=test_data['testing_samples_6']    
        test_attribute_6=test_data['testing_attribute_6']

        test_sample_7=test_data['testing_samples_7']
        test_attribute_7=test_data['testing_attribute_7']

        test_sample_8=test_data['testing_samples_8']
        test_attribute_8=test_data['testing_attribute_8']

        test_sample_9=test_data['testing_samples_9']
        test_attribute_9=test_data['testing_attribute_9']

        test_sample_10=test_data['testing_samples_10']
        test_attribute_10=test_data['testing_attribute_10']

        test_sample_11=test_data['testing_samples_11']
        test_attribute_11=test_data['testing_attribute_11']

        test_sample_12=test_data['testing_samples_12']
        test_attribute_12=test_data['testing_attribute_12']

        test_sample_13=test_data['testing_samples_13']
        test_attribute_13=test_data['testing_attribute_13']

        test_sample_14=test_data['testing_samples_14']
        test_attribute_14=test_data['testing_attribute_14']

        test_sample_15=test_data['testing_samples_15']
        test_attribute_15=test_data['testing_attribute_15']

        

        train_X=np.concatenate([#train_sample_1, 
                        train_sample_2,
                        train_sample_3,
                        train_sample_4,
                        train_sample_5,
                        #train_sample_6,
                        train_sample_7,
                        train_sample_8,
                        train_sample_9,
                        train_sample_10,
                        train_sample_11,
                        train_sample_12,
                        train_sample_13,
                        #train_sample_14,
                        train_sample_15 
                        
                        ], axis=0)

        train_Y=np.concatenate([#train_attribute_1, 
                        train_attribute_2,
                        train_attribute_3,
                        train_attribute_4,
                        train_attribute_5,
                        #train_attribute_6,
                        train_attribute_7,
                        train_attribute_8,
                        train_attribute_9,
                        train_attribute_10,
                        train_attribute_11,
                        train_attribute_12,
                        train_attribute_13,
                        #train_attribute_14,
                        train_attribute_15
                        
                        ], axis=0)
        
        test_X=np.concatenate([test_sample_1, 
                        #test_sample_2,
                        #test_sample_3,
                        #test_sample_4,
                        #test_sample_5,
                        test_sample_6,
                        #test_sample_7,
                        #test_sample_8,
                        #test_sample_9,
                        #test_sample_10,
                        #test_sample_11,
                        #test_sample_12,
                        #test_sample_13,
                        test_sample_14
                        #test_sample_15 
                        
                        ], axis=0)
        
        test_Y=np.concatenate([test_attribute_1, 
                        #test_attribute_2,
                        #test_attribute_3,
                        #test_attribute_4,
                        #test_attribute_5,
                        test_attribute_6,
                        #test_attribute_7,
                        #test_attribute_8,
                        #test_attribute_9,
                        #test_attribute_10,
                        #test_attribute_11,
                        #test_attribute_12,
                        #test_attribute_13,
                        test_attribute_14
                        #test_attribute_15
                        
                        ], axis=0)
        
        traindata=train_X
        train_attributelabel=train_Y
        
        testdata=test_X
        test_attributelabel=test_Y
       
        num_batches=int(traindata.shape[0]/batch_size)
               
        for epoch in range(epochs):
            
            for batch_i in range(num_batches):
                
                start_i =batch_i * batch_size
                end_i=(batch_i + 1) * batch_size
                
                train_x=traindata[start_i:end_i]
                train_y=train_attributelabel[start_i:end_i] 
                                                                               
                self.autoencoder.trainable = True
                self.c.trainable = False
                self.m.trainable = False
                self.d.trainable = False
                self.g.trainable = False
                
                with tf.GradientTape(persistent=True) as tape_auto:
                  feature, output_sample=self.autoencoder(train_x)
                  autoencoder_loss=mean_squared_error(train_x,output_sample)                
                average_autoencoder_loss = tf.reduce_mean(autoencoder_loss)
                grads_autoencoder = tape_auto.gradient(autoencoder_loss, self.autoencoder.trainable_weights)
                self.autoencoder_optimizer.apply_gradients(zip(grads_autoencoder, self.autoencoder.trainable_weights))

                del tape_auto
                         
                self.autoencoder.trainable = False
                self.c.trainable = True
                self.m.trainable = False
                self.d.trainable = False
                self.g.trainable = False

                with tf.GradientTape(persistent=True) as tape_c:
                  feature_c, output_sample_c=self.autoencoder(train_x)
                  hidden_ouput_c,predict_attribute_c=self.c(feature_c)
                  c_loss=self.classification_loss(feature_c,train_y, hidden_ouput_c, predict_attribute_c)
                average_c_loss = tf.reduce_mean(c_loss)
                grads_c = tape_c.gradient(c_loss, self.c.trainable_weights)  
                self.c_optimizer.apply_gradients(zip(grads_c, self.c.trainable_weights))
                
                del tape_c
               
                self.autoencoder.trainable = False
                self.c.trainable = False
                self.m.trainable = True
                self.d.trainable = False
                self.g.trainable = False

                m_loss_1= self.m_model.train_on_batch([train_x,train_x],[similar_truth])

                position=int(start_i/480)
                delete_start=position*480
                delete_end=(position+1)*480

                new_traindata=np.delete(traindata, slice(delete_start, delete_end), axis=0)
                new_train_attribute_label=np.delete(train_attributelabel, slice(delete_start, delete_end), axis=0)
                random_indices = np.random.choice(new_traindata.shape[0], size=batch_size, replace=False)
                selected_train_x= new_traindata[random_indices]             

                m_loss_2= self.m_model.train_on_batch([train_x,selected_train_x],[unsimilar_truth])

                self.autoencoder.trainable = False
                self.c.trainable = False
                self.m.trainable = False
                self.d.trainable = True
                self.g.trainable = False


                for _ in range(self.n_critic):

                  with tf.GradientTape(persistent=True) as tape_d:

                    Noise_shape=(batch_size, 50, 1)
                    noise = tf.random.normal(shape=Noise_shape)
                    fake_feature=self.g([noise,train_y])

                    real_feature, decoded= self.autoencoder(train_x)

                    interpolated_feature = RandomWeightedAverage()([real_feature, fake_feature])
        
                    real_validity =self.d([real_feature,train_y])
                    fake_validity =self.d([fake_feature,train_y])  
                    interploted_validity=self.d([interpolated_feature,train_y])                       
        
                    d_loss_real = self.wasserstein_loss(valid, real_validity)
                    d_loss_fake = self.wasserstein_loss(fake, fake_validity)

                    gradients = tape_d.gradient(interploted_validity, interpolated_feature)

                    gradient_penalty= self.gradient_penalty_loss(gradients)
                                  

                    d_loss = d_loss_real + d_loss_fake +self.LAMBDA_GP * gradient_penalty
                  
                  average_d_loss = tf.reduce_mean(d_loss)

                  grads_d = tape_d.gradient(d_loss, self.d.trainable_weights)

                  self.d_optimizer.apply_gradients(zip(grads_d, self.d.trainable_weights))

                  del tape_d

                self.autoencoder.trainable = False
                self.c.trainable = False
                self.m.trainable = False
                self.d.trainable = False               
                self.g.trainable = True
                
                Noise_shape_g=(batch_size, self.latent_dim, 1)
                noise_g = tf.random.normal(shape=Noise_shape_g)
          
                base_indices = np.random.choice(480, size=batch_size, replace=False)
                all_indices = base_indices[None,:] + 480 * np.arange(11)[:,None]                        

                homogeneous_feature, new_sample_1=self.autoencoder(train_x)
          
                heterogeneous_features, new_samples = zip(*[
                  self.autoencoder(new_traindata[indices]) 
                  for indices in all_indices])
                            
                heterogeneous_attributes = [new_train_attribute_label[indices] for indices in all_indices]
                
                with tf.GradientTape(persistent=True) as tape_g:
            
                  Fake_feature_g=self.g([noise_g,train_y])
                  Fake_validity_g=self.d([Fake_feature_g,train_y])
                  adversarial_loss=self.wasserstein_loss(valid, Fake_validity_g) #L_adv
            
                  fake_hidden_ouput_g,Fake_classification_g= self.c(Fake_feature_g)
                  classification_loss=self.classification_loss(Fake_feature_g,train_y,\
                  fake_hidden_ouput_g,Fake_classification_g) #L_cla
                  
                  comparison_loss_simi=tf.keras.losses.binary_crossentropy(
                  similar_truth, self.m([Fake_feature_g,homogeneous_feature]))
                  
                  comparison_loss_unsimi = tf.reduce_sum([
                    tf.keras.losses.binary_crossentropy(
                        unsimilar_truth,
                        self.m([Fake_feature_g, heterogeneous_features[i]])
                    ) for i in range(11)
                  ],axis=0)
                  
                  cycle_rank_loss = 0
                  
                  if self.crl == True:
                                      
                    reconstructed_feature = self.g([noise_g,Fake_classification_g])
                    Fake_feature_g_unsimi = [
                    self.g([noise_g, heterogeneous_attributes[i]]) 
                    for i in range(11)
                    ]
            
                    cycle_rank_losses = [
                      self.cycle_rank_loss(
                      Fake_feature_g,
                      reconstructed_feature, 
                      Fake_feature_g_unsimi[i],
                      similar_truth                           
                      ) for i in range(11)
                    ]  
                                                    
                    cycle_rank_loss = tf.reduce_mean(cycle_rank_losses,axis=0)   
                           
                  total_loss=adversarial_loss+self.lambda_cla*classification_loss\
                  +self.lambda_cms*comparison_loss_simi+self.lambda_cms*comparison_loss_unsimi\
                  +self.lambda_crl*cycle_rank_loss  
                          
                average_g_loss = tf.reduce_mean(total_loss)  
                grads_g = tape_g.gradient(total_loss, self.g.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grads_g, self.g.trainable_weights))

                del tape_g
                
                elapsed_time = datetime.datetime.now() - start_time
          
                print ("[Epoch %d/%d][Batch %d/%d][Autoencoder loss: %f][C loss: %f][M loss: %f][D loss: %f][G loss %05f ]time: %s " \
                 % (epoch, epochs,
                   batch_i, num_batches,
                   average_autoencoder_loss, 
                     average_c_loss,
                     (m_loss_1+m_loss_2)/2,
                       average_d_loss,
                       average_g_loss,                                                                                                              
                       elapsed_time))
        
            if epoch % 1 == 0:
                       
                accuracy_lsvm,accuracy_nrf,accuracy_pnb,accuracy_mlp=feature_generation_and_diagnosis(2000,testdata,test_attributelabel,gan.autoencoder,gan.g, gan.c)  

                accuracy_list_1.append(accuracy_lsvm) 
                accuracy_list_2.append(accuracy_nrf) 
                accuracy_list_3.append(accuracy_pnb)
                accuracy_list_4.append(accuracy_mlp)

                print("[Epoch %d/%d] [Accuracy_lsvm: %f] [Accuracy_nrf: %f] [Accuracy_pnb: %f][Accuracy_mlp: %f]"\
                  %(epoch, epochs,max(accuracy_list_1),max(accuracy_list_2),max(accuracy_list_3),max(accuracy_list_4)))
            
        print('finished! best_acc:{:.4f}'.format(max([max(accuracy_list_1),max(accuracy_list_2),max(accuracy_list_3),max(accuracy_list_4)])))
                

      
        
if __name__ == '__main__':
    gan = Zero_shot()
    gan.train(epochs=2000, batch_size=120)
