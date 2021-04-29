import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from functools import reduce
from tensorflow import keras

# import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPool2D, BatchNormalization, LSTM, GRU
from tensorflow.keras.layers import Reshape, Permute, Lambda, Bidirectional
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import layers
from ACR_Training.efficient_net import EfficientNetB0

class CustumBatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(CustumBatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        # return super.call, but set training
        if not training:
            return super(CustumBatchNormalization, self).call(inputs, training=False)
        else:
            return super(CustumBatchNormalization, self).call(inputs, training=(not self.freeze))

    def get_config(self):
        config = super(CustumBatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    f1 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=False, name='{}_dconv'.format(name))
    f2 = CustumBatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=False, name='{}_conv'.format(name))
    f2 = CustumBatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))




class RecogBaseModel():
    def __init__(self,weights_path: str = None,backbone_name='vgg'):
        self.weights_path=weights_path
        self.backbone_name = backbone_name


    def upconv(self, x, n, filters):
        x = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=f'upconv{n}.conv.0')(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.1')(x)
        x = keras.layers.Activation('relu', name=f'upconv{n}.conv.2')(x)
        x = keras.layers.Conv2D(filters=filters // 2,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                name=f'upconv{n}.conv.3')(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.4')(x)
        x = keras.layers.Activation('relu', name=f'upconv{n}.conv.5')(x)
        return x
    
    def build_efficientnet_backbone(self, inputs, backbone_name, imagenet):
        backbone = getattr(efficientnet, backbone_name)(include_top=False,
                                                        input_tensor=inputs,
                                                        weights=None)#'imagenet' if imagenet else None)
        return [
            backbone.get_layer(slice_name).output for slice_name in [
                'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
                'block5a_expand_activation'
            ]
        ]



    def build_keras_model(self, inputs):#, weights_path: str = None, backbone_name='vgg'):
        weights_path= self.weights_path
        backbone_name = self.backbone_name
        #inputs = tf.keras.layers.Input((None, None, 3))
        #inputs =
        #print(inputs.shape)

        if backbone_name == 'vgg':
            print("no vgg..please add code.")
            pass
            #s1, s2, s3, s4 = self.build_vgg_backbone(inputs)
        elif 'efficientnet' in backbone_name.lower():
            s1, s2, s3, s4 = self.build_efficientnet_backbone(inputs=inputs,
                                                         backbone_name=backbone_name,
                                                         imagenet=None)#weights_path is None)
        else:
            raise NotImplementedError
            
        s1 = keras.layers.Conv2D(filters=int(s1.shape[-1]), kernel_size=1, strides=1)(s1)
        s2 = keras.layers.Conv2D(filters=int(s1.shape[-1]), kernel_size=1, strides=1)(s2)
        s3 = keras.layers.Conv2D(filters=int(s1.shape[-1]), kernel_size=1, strides=1)(s3)
        s4 = keras.layers.Conv2D(filters=int(s1.shape[-1]), kernel_size=1, strides=1)(s4)
        
        def bifpn_layer(x1,x2,x3,x4, ids=0,just_up=False):
            # upsample
            x4_U = layers.UpSampling2D()(x4)
            P3_td = layers.Add()([x4_U, x3])
            P3_td = layers.Activation('swish')(P3_td)  
            P3_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_U_P3'.format(ids))(P3_td)

            x3_U = layers.UpSampling2D()(P3_td)
            P2_td = layers.Add()([x3_U,x2])
            P2_td = layers.Activation('swish')(P2_td)  
            P2_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_U_P2'.format(ids))(P2_td)

            x2_U = layers.UpSampling2D()(P2_td)
            P1_td = layers.Add()([x2_U,x1])
            P1_td = layers.Activation('swish')(P1_td)  
            P1_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_U_P1'.format(ids))(P1_td)

            #print("P3_td.shape:{},P2_td.shape:{},P1_td.shape:{} , P1_out:{}".format(P3_td.shape,P2_td.shape,P1_td.shape,P1_out.shape))

            if just_up:
                return P1_out, None, None, None
            else:
                # downsample
                P1_D = layers.MaxPooling2D(strides=(2, 2))(P1_out)
                P2_out = layers.Add()([P1_D, P2_td, x2])
                P2_out = layers.Activation('swish')(P2_out)  
                P2_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_D_P1'.format(ids))(P2_out)


                P2_D = layers.MaxPooling2D(strides=(2, 2))(P2_out)
                P3_out = layers.Add()([P2_D,P3_td, x3])
                P3_out = layers.Activation('swish')(P3_out)  
                P3_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_D_P2'.format(ids))(P3_out)

                P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
                P4_out = layers.Add()([P3_D, x4])
                P4_out = layers.Activation('swish')(P4_out)  
                P4_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=False, name='BiFPN_{}_D_P3'.format(ids))(P4_out)

                return  P1_out, P2_out, P3_out, P4_out
            
        s1,s2,s3,s4 = bifpn_layer(s1,s2,s3,s4,ids=0, just_up=False)
        s1,s2,s3,s4 = bifpn_layer(s1,s2,s3,s4,ids=1, just_up=False)
        y,_,_,_ = bifpn_layer(s1,s2,s3,s4,ids=2, just_up=True)
 
        return y






class CTCLayer(layers.Layer):
    def __init__(self, name=None, focal_ctc_on=False,  alpha=1, gamma=0.99):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        self.focal_ctc_on = focal_ctc_on
        self.alpha= alpha
        self.gamma =gamma

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        
        print("tst ctc loss")
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        ctc_loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        
        if self.focal_ctc_on:
            print("use focal_ctc_on")
            p = tf.exp(-ctc_loss)
            focal_ctc_loss = self.alpha*tf.pow((1-p), self.gamma)*ctc_loss
            self.add_loss(focal_ctc_loss)

            # At test time, just return the computed predictions
            
        else:
            self.add_loss(ctc_loss)
        return y_pred



class Attention(tf.keras.Model):

    def __init__(self, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = BahdanauAttentionCell(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = Dense(num_classes)
        
    def call(self, x, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H = x=lstm: contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        
        batch_size = tf.shape(x)[0]
        num_steps = batch_max_length # +1 for [s] at end of sentence.

        #output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = ( tf.fill([batch_size, self.hidden_size],  np.float32(0)), 
                  tf.fill([batch_size, self.hidden_size],  np.float32(0)))
        
        
        if is_train:
            for i in range(num_steps):
                char_onehots = tf.one_hot(text[:,i], depth=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, x, char_onehots)
                
                #print("hidden")
                reshape_hidden =tf.expand_dims(hidden[0], axis=1)

                if i==0:
                    output_hiddens = reshape_hidden
                elif i>0:
                    output_hiddens= tf.concat([output_hiddens, reshape_hidden], axis=1)
                    
            probs = self.generator(output_hiddens)
            
        else:
            targets = tf.fill([batch_size,],  np.int32(0)) # [GO] token
            for i in range(num_steps):
                char_onehots = tf.one_hot(targets, depth=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, x, char_onehots)
                
                probs_step= self.generator(hidden[0])
                next_input=tf.math.argmax(probs_step, axis=1)
                targets = next_input
                
                
                reshape_probs =tf.expand_dims(probs_step, axis=1)

                if i==0:
                    probs = reshape_probs
                elif i>0:
                    probs= tf.concat([probs, reshape_probs], axis=1)
           
                    
        return probs
                    
            

    
class BahdanauAttentionCell(tf.keras.Model):
    def __init__(self, hidden_size, num_embeddings):
        """
        you dont need input_size in tensorflow
        num_embeddings: num_class
        """
        super(BahdanauAttentionCell, self).__init__()
        
        self.i2h = Dense(hidden_size, use_bias=False)
        self.h2h = Dense(hidden_size)
        self.score = Dense(1, use_bias=False)
        self.rnn = tf.keras.layers.LSTMCell(hidden_size)
        self.hidden_size = hidden_size
        

    def call(self, prev_hidden, batch_H,  char_onehots): # 단, key와 value는 같음
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj =tf.expand_dims(self.h2h(prev_hidden[0]), 1)
        
        e = self.score(tf.nn.tanh(batch_H_proj + prev_hidden_proj))
        alpha = tf.nn.softmax(e, axis=1) 
        alpha = tf.keras.layers.Permute((2, 1))(alpha) 
        context = tf.matmul(alpha, batch_H)
        
        
        context= tf.squeeze(context, 1)
        concat_context=  tf.concat([context, char_onehots],1)
        
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return tuple(cur_hidden[1]), alpha
    
    

def HeidiTextRecogModel(input_shape, num_classes, batch_max_length, backbone_name='EfficientNetB0',
         prediction_mode='ctc',
                        rnn_mode ='lstm',
         
         prediction_only=False, #gru=False, cnn=False,
         hidden_size=256,
         leaky_alpha=0.1,lstm_drop_rate=0.1,focal_ctc_on=False,alpha=0.75, gamma=0.5):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
    """
    print("-"*20)
    print("input_shape: ",input_shape)
    print("num_classes: ",num_classes)
    print("batch_max_length: ",batch_max_length)
    print("backbone_name: ",backbone_name)
    print("prediction_mode: ",prediction_mode)
    print("prediction_only: ",prediction_only)
    print("rnn_mode: ",rnn_mode)
    
    act = LeakyReLU(alpha=leaky_alpha)
    
    input_img= layers.Input(
        shape=input_shape, name="image", dtype="float32"
    )
    
    
    efficent_model=EfficientNetB0(
            include_top=False,
            input_shape=input_shape,
            weights=None,
            classes=25
        )

    hidden = efficent_model(input_img)
    x = tf.keras.layers.Reshape((batch_max_length, -1))(hidden[-1])

    x = layers.Dense(64, name="dense1")(x)
    x = act(x)
    x = layers.Dropout(0.2)(x)

    if 'cnn' in rnn_mode:
        for i in range(6):
            x = BatchNormalization()(x)
            x1 = Conv1D(128, 5, strides=1, dilation_rate=1, padding='same')(x)
            x = act(x)
            x2 = Conv1D(128, 5, strides=1, dilation_rate=2, padding='same')(x)
            x = act(x)
            x = concatenate([x1,x2])
    elif 'gru' in rnn_mode:
        x = Bidirectional(GRU(128,dropout=lstm_drop_rate,recurrent_dropout=lstm_drop_rate, return_sequences=True, reset_after=False))(x)
        x = act(x)
        x = Bidirectional(GRU(128,dropout=lstm_drop_rate,recurrent_dropout=lstm_drop_rate, return_sequences=True, reset_after=False))(x)
        x = act(x)
        
    elif 'lstm' in rnn_mode:
        x = Bidirectional(LSTM(128,dropout=lstm_drop_rate,recurrent_dropout=lstm_drop_rate, return_sequences=True, name='lstm_1'))(x)
        x = act(x)
        x = Bidirectional(LSTM(128,dropout=lstm_drop_rate,recurrent_dropout=lstm_drop_rate, return_sequences=True, name='lstm_2'))(x)
        x = act(x)
       

    if 'ctc' in prediction_mode :
        labels = layers.Input(name="label", shape=(batch_max_length,), dtype="float32")
        x = Dense(num_classes)(x)
        x  = Activation('softmax', name='softmax')(x)

        model_pred = Model(input_img, x)
        if prediction_only:
            return model_pred

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss",focal_ctc_on=focal_ctc_on,  alpha=alpha, gamma=gamma)(labels, x)

        # Define the model
        model_train = keras.models.Model(
            inputs=[input_img, labels], outputs=[output], name="ocr_model_v1"
        )
        
        print("output:",output.shape)
        
    else:
        labels = layers.Input(name="label", shape=(batch_max_length,), dtype="int32")
        
        #hidden_size = 64
        attention = Attention(hidden_size, num_classes)
        
        
        predict_probs= attention(x=x, text=None, is_train=False, batch_max_length=batch_max_length)
        predict_x  = Activation('softmax', name='softmax_p')(predict_probs)
        model_pred = keras.models.Model([input_img], predict_x)
        
        if prediction_only:
            return model_pred
        
        train_probs= attention(x=x, text=labels, is_train=True, batch_max_length=batch_max_length)
        train_x  = Activation('softmax', name='softmax_t')(train_probs)
        
        model_train = keras.models.Model([input_img,labels], train_x, name="ocr_model_v1")
        
        
        print("output:",train_x.shape)
        
    return model_train, model_pred





def heidi_loss(y_true_, y_pred_):
    cce=tf.keras.losses.CategoricalCrossentropy()
    all_loss=0
    for yi in range(0,y_pred_.shape[1]):
        y_true= y_true_[:,yi,:]
        y_pred= y_pred_[:,yi,:]
        
        all_loss+= cce(y_true,y_pred)
        
    return all_loss

def heidi_acc(y_true_, y_pred_):
    all_acc=0
    for yi in range(0,y_pred_.shape[1]):
        y_true= y_true_[:,yi,:]
        y_pred= y_pred_[:,yi,:]
        
        acc1 = K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

        all_acc+=acc1
    return ((all_acc)/y_pred_.shape[1])

def catg_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=[2], keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, [1,2])
    return loss
# def loss_fn(ytrue, ypred):
#     return -K.sum(ytrue*K.log(ypred+1e-6), [1,2])



def catg_acc(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def soft_acc(y_true, y_pred):
       return K.mean(K.equal(K.round(y_true), K.round(y_pred)))