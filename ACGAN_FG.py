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
        self.n_critic = 1
        self.LAMBDA_GP=10
        self.num_blocks=3
        self.crl = False

        self.lambda_adv = 1
        self.lambda_class = 10
        self.lambda_silimar = 10
        self.lambda_unsilimar = 10
        
        self.bound = False
        self.mi_weight = 0.001
        self.mi_bound = 100
        
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)        
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.m_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
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
        
        self.autoencoder.trainable = False
        self.c.trainable = False
        self.m.trainable = False
        self.d.trainable = False
        self.g.trainable = True
        
        
        noise= Input(shape=self.noise_shape)
        attribute = Input(shape=(20,1), dtype='int32')

        homogeneous_sample=Input(shape=self.sample_shape)

        heterogeneous_sample_1=Input(shape=self.sample_shape)
        heterogeneous_sample_2=Input(shape=self.sample_shape)
        heterogeneous_sample_3=Input(shape=self.sample_shape)
        heterogeneous_sample_4=Input(shape=self.sample_shape)
        heterogeneous_sample_5=Input(shape=self.sample_shape)
        heterogeneous_sample_6=Input(shape=self.sample_shape)
        heterogeneous_sample_7=Input(shape=self.sample_shape)
        heterogeneous_sample_8=Input(shape=self.sample_shape)
        heterogeneous_sample_9=Input(shape=self.sample_shape)
        heterogeneous_sample_10=Input(shape=self.sample_shape)
        heterogeneous_sample_11=Input(shape=self.sample_shape)

        homogeneous_feature, new_sample_1=self.autoencoder(homogeneous_sample)

        heterogeneous_feature_1, new_sample_2_1=self.autoencoder(heterogeneous_sample_1)
        heterogeneous_feature_2, new_sample_2_2=self.autoencoder(heterogeneous_sample_2)
        heterogeneous_feature_3, new_sample_2_3=self.autoencoder(heterogeneous_sample_3)
        heterogeneous_feature_4, new_sample_2_4=self.autoencoder(heterogeneous_sample_4)
        heterogeneous_feature_5, new_sample_2_5=self.autoencoder(heterogeneous_sample_5)
        heterogeneous_feature_6, new_sample_2_6=self.autoencoder(heterogeneous_sample_6)
        heterogeneous_feature_7, new_sample_2_7=self.autoencoder(heterogeneous_sample_7)
        heterogeneous_feature_8, new_sample_2_8=self.autoencoder(heterogeneous_sample_8)
        heterogeneous_feature_9, new_sample_2_9=self.autoencoder(heterogeneous_sample_9)
        heterogeneous_feature_10, new_sample_2_10=self.autoencoder(heterogeneous_sample_10)
        heterogeneous_feature_11, new_sample_2_11=self.autoencoder(heterogeneous_sample_11)
        

        Fake_feature=self.g([noise,attribute])
        
        Fake_validity=self.d([Fake_feature,attribute])

        fake_hidden_ouput,Fake_classification= self.c(Fake_feature)

        predicted_similarity_similar=self.m([Fake_feature,homogeneous_feature])

        predicted_similarity_unsimilar_1=self.m([Fake_feature,heterogeneous_feature_1])
        predicted_similarity_unsimilar_2=self.m([Fake_feature,heterogeneous_feature_2])
        predicted_similarity_unsimilar_3=self.m([Fake_feature,heterogeneous_feature_3])
        predicted_similarity_unsimilar_4=self.m([Fake_feature,heterogeneous_feature_4])
        predicted_similarity_unsimilar_5=self.m([Fake_feature,heterogeneous_feature_5])
        predicted_similarity_unsimilar_6=self.m([Fake_feature,heterogeneous_feature_6])
        predicted_similarity_unsimilar_7=self.m([Fake_feature,heterogeneous_feature_7])
        predicted_similarity_unsimilar_8=self.m([Fake_feature,heterogeneous_feature_8])
        predicted_similarity_unsimilar_9=self.m([Fake_feature,heterogeneous_feature_9])
        predicted_similarity_unsimilar_10=self.m([Fake_feature,heterogeneous_feature_10])
        predicted_similarity_unsimilar_11=self.m([Fake_feature,heterogeneous_feature_11])


      
        self.g_model = Model(inputs=[noise,attribute,homogeneous_sample,heterogeneous_sample_1,heterogeneous_sample_2,heterogeneous_sample_3,heterogeneous_sample_4,heterogeneous_sample_5
                          ,heterogeneous_sample_6,heterogeneous_sample_7,heterogeneous_sample_8,heterogeneous_sample_9,heterogeneous_sample_10,heterogeneous_sample_11],
                  outputs=[Fake_validity,Fake_classification,predicted_similarity_similar,predicted_similarity_unsimilar_1,predicted_similarity_unsimilar_2,predicted_similarity_unsimilar_3,
                      predicted_similarity_unsimilar_4,predicted_similarity_unsimilar_5,predicted_similarity_unsimilar_6,predicted_similarity_unsimilar_7,predicted_similarity_unsimilar_8,
                      predicted_similarity_unsimilar_9,predicted_similarity_unsimilar_10,predicted_similarity_unsimilar_11])
        
        self.g_model.compile(loss=[self.wasserstein_loss, 'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
                    'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'],
                  loss_weights=[ self.lambda_adv, self.lambda_class, self.lambda_silimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,
                          self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar,self.lambda_unsilimar],
                                        optimizer=self.g_optimizer)
             

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
        
        PATH_train='/home/liaowenjie/myfolder/GAN_for_UFD_3/dataset_train_case1.npz'
        PATH_test='/home/liaowenjie/myfolder/GAN_for_UFD_3/dataset_test_case1.npz'
        
        
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
                
                random_indices_1 = np.random.choice(480, size=batch_size, replace=False)
                random_indices_2 = random_indices_1+480*1
                random_indices_3 = random_indices_1+480*2
                random_indices_4 = random_indices_1+480*3
                random_indices_5 = random_indices_1+480*4
                random_indices_6 = random_indices_1+480*5
                random_indices_7 = random_indices_1+480*6
                random_indices_8 = random_indices_1+480*7
                random_indices_9 = random_indices_1+480*8
                random_indices_10 = random_indices_1+480*9
                random_indices_11 = random_indices_1+480*10
                selected_train_x_1= new_traindata[random_indices_1]
                selected_train_x_2= new_traindata[random_indices_2]
                selected_train_x_3= new_traindata[random_indices_3]
                selected_train_x_4= new_traindata[random_indices_4]
                selected_train_x_5= new_traindata[random_indices_5]
                selected_train_x_6= new_traindata[random_indices_6]
                selected_train_x_7= new_traindata[random_indices_7]
                selected_train_x_8= new_traindata[random_indices_8]
                selected_train_x_9= new_traindata[random_indices_9]
                selected_train_x_10= new_traindata[random_indices_10]
                selected_train_x_11= new_traindata[random_indices_11]
                
                Noise_shape_g=(batch_size, self.latent_dim, 1)
                noise_g = tf.random.normal(shape=Noise_shape_g)
          
                g_loss= self.g_model.train_on_batch([noise_g,train_y, train_x,selected_train_x_1,selected_train_x_2,selected_train_x_3,selected_train_x_4,
                                  selected_train_x_5,selected_train_x_6,selected_train_x_7,selected_train_x_8,selected_train_x_9,selected_train_x_10,selected_train_x_11],
                    [valid,train_y,similar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth,
                    unsimilar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth,unsimilar_truth])                              
                
                elapsed_time = datetime.datetime.now() - start_time
          
                print ("[Epoch %d/%d][Batch %d/%d][Autoencoder loss: %f][C loss: %f][M loss: %f][D loss: %f][G loss %05f ]time: %s " \
                 % (epoch, epochs,
                   batch_i, num_batches,
                   average_autoencoder_loss, 
                     average_c_loss,
                     (m_loss_1+m_loss_2)/2,
                       average_d_loss,
                       g_loss[0],                                                                                                              
                       elapsed_time))
        
            if epoch % 10 == 0:
                       
                accuracy_lsvm,accuracy_nrf,accuracy_pnb,accuracy_mlp=feature_generation_and_diagnosis(2000,testdata,test_attributelabel,gan.autoencoder,gan.g, gan.c)              

                print("[Epoch %d/%d] [Accuracy_lsvm: %f] [Accuracy_nrf: %f] [Accuracy_pnb: %f][Accuracy_mlp: %f]"\
                  %(epoch, epochs,accuracy_lsvm,accuracy_nrf,accuracy_pnb,accuracy_mlp))
            
                accuracy_list_1.append(accuracy_lsvm) 
                accuracy_list_2.append(accuracy_nrf) 
                accuracy_list_3.append(accuracy_pnb)
                accuracy_list_4.append(accuracy_mlp)            

                print(accuracy_list_1)
                print(accuracy_list_2)
                print(accuracy_list_3)
                print(accuracy_list_4)

      
        
if __name__ == '__main__':
    gan = Zero_shot()
    gan.train(epochs=1000, batch_size=60)
