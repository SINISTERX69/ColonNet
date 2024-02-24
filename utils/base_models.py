from keras.models import Model
from keras.layers import * 
from keras.applications import DenseNet121



def build_model():
    sp=(224,224,3)
    input_layer=Input(shape=sp,name="Input")

    base =DenseNet121(include_top=False,input_tensor= input_layer,input_shape=sp)(input_layer)

    #Classification branch
    f1=MaxPooling2D(name='pool')(base)
    class1=Dense(1024,activation='relu',name="c_1")(f1)
    d1=Dropout(0.3)(class1)
    d1=Flatten()(d1)
    class2=Dense(512,activation='relu',name="c_2")(d1)
    d2=Dropout(0.2)(class2)
    class4=Dense(128,activation='relu',name="c_4")(d2)
    class5=Dense(64,activation='relu',name="c_5")(class4)
    class6=Dense(1,activation='sigmoid',name="c_final")(class5)


    #Bounding Box Branch
    f2=GlobalMaxPool2D()(base)
    regress1=Dense(2500,activation='relu',name='b_1')(f2)
    regress2=Dense(1500,activation='relu',name='b_2')(regress1)
    regress3=Dense(750,activation='ELU',name='b_3')(regress2)
    regress4=Dense(512,activation="relu",name='b_4')(regress3)
    r2=Dropout(0.3)(regress4)
    regress5=Dense(256,activation='ELU',name='b_5')(r2)
    regress6=Dense(4,activation='sigmoid',name='b_final')(regress5)

    mdl=Model(inputs=input_layer,outputs=[class6,regress6],name="ColonSeg")
    return mdl


def double_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def Build_Unet_Model(num_filters=64,input_shape=(224,224,3)):
    inputs = Input(input_shape)

    #Encoder Path
    f_1 = double_conv_block(inputs, num_filters*1)
    p_1 = MaxPool2D((2, 2))(f_1)
    f_2 = double_conv_block(p_1, num_filters*2)
    p_2 = MaxPool2D((2, 2))(f_2)
    f_3 = double_conv_block(p_2, num_filters*4)
    p_3 = MaxPool2D((2, 2))(f_3)
    f_4 = double_conv_block(p_3, num_filters*8)
    p_4 = MaxPool2D((2, 2))(f_4)

    c   = double_conv_block(p_4,num_filters*16)

    # Decoder path
    d_1 = Conv2DTranspose(num_filters*8, 2, strides=2, padding="same")(c)
    d_1 = Concatenate()([d_1, f_4])
    d_1 = double_conv_block(d_1, num_filters*8)

    d_2 = Conv2DTranspose(num_filters*4, 2, strides=2, padding="same")(d_1)
    d_2 = Concatenate()([d_2, f_3])
    d_2 = double_conv_block(d_2, num_filters*4)

    d_3 = Conv2DTranspose(num_filters*2, 2, strides=2, padding="same")(d_2)
    d_3 = Concatenate()([d_3, f_2])
    d_3 = double_conv_block(d_3, num_filters*2)

    d_4 = Conv2DTranspose(num_filters*1, 2, strides=2, padding="same")(d_3)
    d_4 = Concatenate()([d_4, f_1])
    d_4 = double_conv_block(d_4, num_filters*1)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d_4)
    Unet_Model = Model(inputs, outputs, name="UNetModel")
    return Unet_Model