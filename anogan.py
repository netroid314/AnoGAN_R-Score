from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as tf_back
import numpy as np

def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def discriminator_model():
    inputs = Input((28, 28, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(10,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    return gan

def load_model(target_label = "1"):
    d = discriminator_model()
    g = generator_model()

    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)

    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    d.load_weights('./weights/discriminator_'+target_label+'.h5')
    g.load_weights('./weights/generator_'+target_label+'.h5')

    return g, d

def train(BATCH_SIZE, X_train, target_label = 'default'):
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.0004)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    
    for epoch in range(20):
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        
        for index in range(n_iter):
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))

            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            d_loss = d.train_on_batch(X, y)

            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

        g.save_weights('weights/generator_'+target_label+'.h5', True)
        d.save_weights('weights/discriminator_'+target_label+'.h5', True)
    return d, g

def generate(BATCH_SIZE, target_label = "-1"):
    g = generator_model()

    g.load_weights('weights/generator_'+target_label+'.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
    generated_images = g.predict(noise)
    return generated_images

def sum_of_residual(y_true, y_pred):
    return tf_back.sum(tf_back.abs(y_true - y_pred))

def feature_extractor(d_label=None):
    d = discriminator_model()
    d.load_weights('weights/discriminator_'+d_label+'.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

def anomaly_detector(g_label=None, d_label=None):
    g = generator_model()
    g.load_weights('weights/generator_'+g_label+'.h5')
    intermidiate_model = feature_extractor(d_label)
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    
    tf_back.set_learning_phase(0)
    
    return model

def compute_anomaly_score(model, x, iterations=500, d_label=None):
    z = np.random.uniform(0, 1, size=(1, 10))
    
    intermidiate_model = feature_extractor(d_label)
    d_x = intermidiate_model.predict(x)

    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)

    key_list = list(loss.history.keys())
    discrimitive_loss = loss.history[key_list[2]][-1]
    residual_loss = loss.history[key_list[1]][-1]

    return residual_loss, discrimitive_loss