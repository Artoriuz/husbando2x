# HUSBANDO_2X RCAN 2RG 5RCAB

import numpy as np
import tensorflow as tf
import cv2
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def ssim_metric(y_true, y_pred):
    ssim_1 = tf.image.ssim(y_true, y_pred, max_val=255)
    return tf.reduce_mean(ssim_1)

# Start of the Model
inputs = tf.keras.Input(shape=(None,None,3))
scaled_inputs = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)(inputs)
scaled_inputs = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(scaled_inputs)

# 1st RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(scaled_inputs)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab1 = tf.keras.layers.Add()([multi, scaled_inputs])

# 2nd RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab1)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab2 = tf.keras.layers.Add()([multi, rcab1])

# 3rd RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab2)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab3 = tf.keras.layers.Add()([multi, rcab2])

# 4th RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab3)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab4 = tf.keras.layers.Add()([multi, rcab3])

# 5th RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab4)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab5 = tf.keras.layers.Add()([multi, rcab4])

# 1st RG
rg1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab5)
rg1 = tf.keras.layers.Add()([rg1, scaled_inputs])

# 1st RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rg1)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab1 = tf.keras.layers.Add()([multi, scaled_inputs])

# 2nd RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab1)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab2 = tf.keras.layers.Add()([multi, rcab1])

# 3rd RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab2)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab3 = tf.keras.layers.Add()([multi, rcab2])

# 4th RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab3)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab4 = tf.keras.layers.Add()([multi, rcab3])

# 5th RCAB
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab4)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
avg_pool = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
feat_mix = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation='relu')(avg_pool)
feat_mix = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(feat_mix)
multi = tf.keras.layers.Multiply()([feat_mix, x])
rcab5 = tf.keras.layers.Add()([multi, rcab4])

# 2nd RG
rg2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(rcab5)
rg2 = tf.keras.layers.Add()([rg2, rg1])

# Feature Fusion
add_global = tf.keras.layers.Add()([rg2, scaled_inputs])
features = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(add_global)
high_res = tf.nn.depth_to_space(features, 2)
high_res = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(high_res)
high_res = tf.keras.layers.ReLU(max_value=1)(high_res)
outputs = tf.keras.layers.experimental.preprocessing.Rescaling(255)(high_res)

# Defining the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Defining Adam1
Adam1 = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compiling the model
model.compile(optimizer=Adam1, loss='mae', metrics=ssim_metric)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Load the training data
filelist1 = sorted(glob.glob('./quarter_split/*.png'))
train_in = []
for myFile in filelist1:
    image = cv2.imread(myFile, cv2.IMREAD_COLOR)
    train_in.append (image)
train_in = np.array(train_in).astype(np.float32)

filelist2 = sorted(glob.glob('./half_split/*.png'))
train_ref = []
for myFile in filelist2:
    image = cv2.imread(myFile, cv2.IMREAD_COLOR)
    train_ref.append (image)
train_ref = np.array(train_ref).astype(np.float32)

# Create checkpoints
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoint/', save_weights_only=True, monitor='loss', mode='min', save_best_only=True)

# Train the model
history = model.fit(train_in, train_ref, epochs=100, batch_size=64, verbose=1, callbacks=[model_checkpoint_callback])

# Save model
model.save('./model.h5')
model.save('./model/')

# Load weights
model.load_weights('./model.h5')

# Make a single prediction
image = cv2.imread('./input.png', cv2.IMREAD_COLOR)
image = image.astype(np.float32).reshape(1,540,960,3) # reshaping is needed to add the first dimension

predictions = model.predict(image)
predictions = np.squeeze((np.around(predictions)).astype(np.uint8))
cv2.imwrite('./prediction.png', predictions)