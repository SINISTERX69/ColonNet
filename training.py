from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf


from utils.losses import *
from utils.data_loaders import *
from utils.base_models import *

# first training the Bounding Box on bleeding images only
X_train,X_val,box_train,box_val,ann_train,ann_val=train_test_split(
    *load_data(with_neg=False,aug=True),test_size=0.2
)

model=build_model()


# turn off learning of Classification branch
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable=False

#defining losses
losses={
    "c_final":tf.keras.losses.BinaryCrossentropy(),
    "b_final":tf.keras.losses.MSE
}  


#defining targets for each branch
train_target={
    "c_final":ann_train,
    "b_final":box_train
}

val_target={
    "c_final":ann_val,
    "b_final":box_val
}

#defining optimizer : AdamW
opt=tf.keras.optimizers.AdamW(learning_rate=1e-4)

#training the bounding-box branch
model.compile(loss=losses,optimizer=opt)
model.fit(X_train,train_target,validation_data=(X_val,val_target),epochs=10,batch_size=64)

model.save("CheckPoint1.h5")

# Training the classification branch on both images now

model=load_model("CheckPoint1.h5")

# Turning off box layers and turning on the classification layers
for layer in model.layers:
    if layer.name.startswith("b_")or layer.name=="densenet121":
        layer.trainable=False
    else:
        layer.trainable=True

#loading and spliting data for training
X_train,X_val,box_train,box_val,ann_train,ann_val=train_test_split(
    *load_data(aug=False,nums=2),test_size=0.2
)


#defining targets for each branch
train_target={
    "c_final":ann_train,
    "b_final":box_train
}

val_target={
    "c_final":ann_val,
    "b_final":box_val
}

#defining optimizer : "AdamW"
opt=tf.keras.optimizers.AdamW(learning_rate=1e-4)


#training classification branch
model.compile(loss=losses,optimizer=opt,metrics=["accuracy"])
model.fit(X_train,train_target,validation_data=(X_val,val_target),epochs=10,batch_size=64)


#saving model for future use
model.save("classNbox.h5")


# Now training the unet model for segmentation
X_train,X_val,y_train,y_val=train_test_split(*load_data_unet(True,2),
                                             test_size=0.2,shuffle=True)
y_train=y_train.reshape(-1,224,224,1)
y_val=y_val.reshape(-1,224,224,1)

model=Build_Unet_Model()
model.compile(optimizer='adam',loss=focal_tversky,metrics=['tversky'])


#callbacks
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min', 
                              verbose=1, 
                              patience=20
                             )
# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="ResUNet-segModel-weights.hdf5", 
                               verbose=1, 
                               save_best_only=True
                              )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                             )


model.fit(X_train,y_train,epochs=30,validation_data=(X_val,y_val),
          callbacks=[earlystopping,checkpointer,reduce_lr])
model.save("segmentation.hdf5")

# Final model creation


upper=load_model("classNbox.h5")
lower=load_model("segmentation.hdf5")

for m in (upper,lower):
    for layer in m:
        layer.trainable=False

input_layer=Input(shape=(224,224,3),name=Input)
l1=upper(input_layer)
l2=lower(input_layer)

comb_out=[*l1,l2]
final=Model(inputs=input_layer,outputs=comb_out,name="ColonNet")

final.save("ColonNet.h5")
