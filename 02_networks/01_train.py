import tensorflow as tf
import os
from networks import DnCNN, DRUNet, Pix2Pix
from dataloader import DatasetLoader
from callbacks import ImageLogger, GANCheckpoint


MODEL_NAME = 'pix2pix'  # select model to train: 'dncnn', 'drunet', 'pix2pix'
GRAYSCALE = False  # set true for grayscale images, false for RGB
CHANNELS = 1 if GRAYSCALE else 3

BATCH_SIZE = 64  # set batch size
EPOCHS = 200  # set number of epochs

BASE_DIR = ''  # set base directory for dataset
TRAIN_CLEAN = os.path.join(BASE_DIR, 'train/clean')
TRAIN_NOISY = os.path.join(BASE_DIR, 'train/degraded')
VAL_CLEAN = os.path.join(BASE_DIR, 'val/clean')
VAL_NOISY = os.path.join(BASE_DIR, 'val/degraded')
TEST_CLEAN = os.path.join(BASE_DIR, 'test/clean')
TEST_NOISY = os.path.join(BASE_DIR, 'test/degraded')


LOG_IMG_DIR = f"./logs/{MODEL_NAME}_images"  # directory to save logged images
CHECKPOINT_DIR = "./checkpoints"  # directory to save model checkpoints


def train():
    # enable memory growth for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled Memory Growth for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)

    print(f"Setting up {MODEL_NAME}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading Data...")
    train_loader = DatasetLoader(TRAIN_CLEAN, TRAIN_NOISY, model_name=MODEL_NAME, grayscale=GRAYSCALE)
    val_loader = DatasetLoader(VAL_CLEAN, VAL_NOISY, model_name=MODEL_NAME, grayscale=GRAYSCALE)
    test_loader = DatasetLoader(TEST_CLEAN, TEST_NOISY, model_name=MODEL_NAME, grayscale=GRAYSCALE)

    train_ds = train_loader.get_dataset(BATCH_SIZE, shuffle=True)
    val_ds = val_loader.get_dataset(BATCH_SIZE, shuffle=False)
    test_ds = test_loader.get_dataset(BATCH_SIZE, shuffle=False)

    print("Building Model...")
    if MODEL_NAME == 'dncnn':
        model = DnCNN(in_nc=CHANNELS, out_nc=CHANNELS)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mean_squared_error')

    elif MODEL_NAME == 'drunet':
        model = DRUNet(in_nc=CHANNELS, out_nc=CHANNELS)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mean_squared_error')

    elif MODEL_NAME == 'pix2pix':
        model = Pix2Pix(in_nc=CHANNELS, out_nc=CHANNELS)
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # define custom loss functions
        def gen_loss(disc_gen, gen_out, target):
            gan_loss = loss_obj(tf.ones_like(disc_gen), disc_gen)
            l1_loss = tf.reduce_mean(tf.abs(target - gen_out))
            return gan_loss + (100 * l1_loss)  # Lambda=100

        def disc_loss(real, gen):
            return loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(gen), gen)

        model.compile(g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                      d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                      gen_loss_fn=gen_loss, disc_loss_fn=disc_loss)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

    # uncomment to see model summary
    # dummy_x = tf.zeros((1, 256, 256, CHANNELS))
    # model(dummy_x)
    # print(model.summary())

    # configure checkpoint based on Model Type
    if MODEL_NAME == 'pix2pix':
        # for pix2pix, save best generator
        # monitor validation generator loss
        monitor_metric = 'val_gen_loss'
        checkpoint_cb = GANCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.weights.h5"),
            monitor=monitor_metric,
            save_best_only=True
        )
    else:
        # for DnCNN and DRUNet, save best model weights
        monitor_metric = 'val_loss'
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.weights.h5"),
            save_best_only=True, save_weights_only=True,
            monitor=monitor_metric
        )

    callbacks = [
        checkpoint_cb,
        tf.keras.callbacks.CSVLogger(f"{MODEL_NAME}_log.csv"),
        ImageLogger(val_ds, LOG_IMG_DIR, model_name=MODEL_NAME, freq=1)
    ]


    # train
    print("Starting training")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("Training complete")


train()
