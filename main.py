import os
import inspect
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Compatibility patch for older voxelmorph with Python 3.11+
if not hasattr(inspect, 'getargspec'):
    def legacy_getargspec(func):
        full = inspect.getfullargspec(func)
        return full.args, full.varargs, full.varkw, full.defaults
    inspect.getargspec = legacy_getargspec

os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import voxelmorph as vxm

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SHAPE   = (112, 112)   # EchoNet-Peds standard frame size
INT_STEPS   = 7
INT_RES     = 2            # integrate at half resolution
LR          = 1e-4
NCC_WEIGHT  = 1.0
GRAD_WEIGHT = 0.5          # higher than default; echo has thin walls

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

model = vxm.networks.VxmDense(
    inshape=IMG_SHAPE,
    nb_unet_features=[enc_nf, dec_nf],
    int_steps=INT_STEPS,
    int_resolution=INT_RES,
)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)
model.compile(
    optimizer=optimizer,
    loss=[vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss],
    loss_weights=[NCC_WEIGHT, GRAD_WEIGHT],
)
print(f"Model: 2D VxmDense {IMG_SHAPE} | TF {tf.__version__}")
model.summary()

# ── Data loading (EchoNet-Peds AVI clips) ────────────────────────────────────
def load_echo_frames(avi_path: str, target_size=IMG_SHAPE):
    """Load an EchoNet-Peds AVI and return all frames as float32 in [0,1]."""
    cap = cv2.VideoCapture(avi_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_size[1], target_size[0]))
        frames.append(resized.astype('float32') / 255.0)
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W)

def get_ed_es_pair(frames: np.ndarray, ed_idx: int, es_idx: int):
    """
    Return (moving, fixed) = (ED frame, ES frame) shaped (1, H, W, 1).
    ED→ES captures systolic contraction — the motion of clinical interest.
    """
    ed = frames[ed_idx][..., np.newaxis][np.newaxis]   # (1, H, W, 1)
    es = frames[es_idx][..., np.newaxis][np.newaxis]   # (1, H, W, 1)
    return ed, es

def make_zero_phi(img_shape, int_resolution):
    """Compute flow target shape dynamically — don't hardcode."""
    flow_shape = tuple(s // int_resolution for s in img_shape) + (len(img_shape),)
    return np.zeros((1,) + flow_shape, dtype='float32')

# ── Training loop with real data ──────────────────────────────────────────────
def train(video_dir: str, file_list: list, ed_es_labels: dict, epochs: int = 50):
    """
    Args:
        video_dir:     path to EchoNet-Peds Videos/ folder
        file_list:     list of AVI filenames
        ed_es_labels:  dict mapping filename → (ed_frame_idx, es_frame_idx)
                       from FileList.csv (columns: ED, ES)
        epochs:        training epochs
    """
    zero_phi = make_zero_phi(IMG_SHAPE, INT_RES)

    for epoch in range(epochs):
        epoch_losses = []
        np.random.shuffle(file_list)

        for fname in file_list:
            path = os.path.join(video_dir, fname)
            if not os.path.exists(path):
                continue

            ed_idx, es_idx = ed_es_labels[fname]
            frames = load_echo_frames(path)

            if ed_idx >= len(frames) or es_idx >= len(frames):
                continue

            moving, fixed = get_ed_es_pair(frames, ed_idx, es_idx)
            loss = model.train_on_batch([moving, fixed], [fixed, zero_phi])
            epoch_losses.append(loss)

        mean_loss = np.mean(epoch_losses, axis=0)
        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"Total: {mean_loss[0]:.4f}  NCC: {mean_loss[1]:.4f}  Grad: {mean_loss[2]:.4f}"
        )

# ── Visualization ──────────────────────────────────────────────
def visualize_pair(moving, fixed, warped=None, title=""):
    """Quick plot of moving / fixed / warped"""
    fig, axs = plt.subplots(1, 3 if warped is not None else 2, figsize=(12, 5))
    axs[0].imshow(moving[0, ..., 0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Moving (ED)')
    axs[1].imshow(fixed[0, ..., 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Fixed (ES)')

    if warped is not None:
        axs[2].imshow(warped[0, ..., 0], cmap='gray', vmin=0, vmax=1)
        axs[2].set_title('Warped (ED - ES)')

    plt.suptitle(title)
    plt.show()

def visualize_flow(flow, title="Flow field"):
    mag = np.sqrt(np.sum(flow**2, axis=-1))
    plt.imshow(mag[0], cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.show()

def train_dummy(epochs=5):
    """Verify shapes and loss components before loading real data."""
    zero_phi = make_zero_phi(IMG_SHAPE, INT_RES)
    print(f"Flow target shape: {zero_phi.shape}")  # should be (1, 56, 56, 2)

    for epoch in range(epochs):
        m = np.random.rand(1, *IMG_SHAPE, 1).astype('float32')
        f = np.random.rand(1, *IMG_SHAPE, 1).astype('float32')
        loss = model.train_on_batch([m, f], [f, zero_phi])
        print(f"  Epoch {epoch+1} — Total: {loss[0]:.4f}  NCC: {loss[1]:.4f}  Grad: {loss[2]:.4f}")

train_dummy()