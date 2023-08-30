import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential

# Generator model
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))
    return model

# Discriminator model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# GAN parameters
latent_dim = 100
input_shape = (28, 28, 1)
output_shape = (28, 28, 1)

# Build and compile models
generator = build_generator(latent_dim, output_shape)
discriminator = build_discriminator(input_shape)
gan = build_gan(generator, discriminator)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Load and preprocess dataset (MNIST for example)
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

# Training loop
epochs = 10000
batch_size = 128

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
    discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
    
    # Train generator
    noise = np.random.randn(batch_size, latent_dim)
    generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, D Loss: {discriminator_loss[0]}, G Loss: {generator_loss}")

        # Save generated images
        if epoch % 1000 == 0:
            samples = generator.predict(np.random.randn(16, latent_dim))
            samples = 0.5 * samples + 0.5  # Rescale images to [0, 1]
            # Save or display the generated images here