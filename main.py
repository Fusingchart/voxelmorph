import os
import inspect
import numpy as np

if not hasattr(inspect, 'getargspec'):
    def legacy_getargspec(func):
        full = inspect.getfullargspec(func)
        return full.args, full.varargs, full.varkw, full.defaults


    inspect.getargspec = legacy_getargspec

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import voxelmorph as vxm

in_shape = (128, 128, 128)
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

model = vxm.networks.VxmDense(
    inshape=in_shape,
    nb_unet_features=[enc_nf, dec_nf],
    int_steps=7,
    int_resolution=2  # Updated from int_downsize to satisfy the warning
)

# Using the legacy optimizer for M1/M2 speed
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=[vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss],
    loss_weights=[1.0, 0.01]
)

print(f"Model initialized with TF {tf.__version__} on Apple Silicon.")

def train_dummy(epochs=5):
    # Dummy data representing 1 moving and 1 fixed volume
    m_dummy = np.random.rand(1, 128, 128, 128, 1).astype('float32')
    f_dummy = np.random.rand(1, 128, 128, 128, 1).astype('float32')

    # Target for flow (half resolution if int_resolution=2)
    zero_phi = np.zeros((1, 64, 64, 64, 3)).astype('float32')

    for epoch in range(epochs):
        # The key fix: capture the loss from the train_on_batch call
        loss = model.train_on_batch([m_dummy, f_dummy], [f_dummy, zero_phi])
        print(f"Epoch {epoch + 1} - Total Loss: {loss[0]:.4f} (Sim: {loss[1]:.4f}, Grad: {loss[2]:.4f})")

train_dummy()