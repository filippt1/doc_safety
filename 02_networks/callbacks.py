import tensorflow as tf
import numpy as np
import cv2
import os


# custom callbacks class for logging images during training
class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, log_dir, model_name='dncnn', freq=1):
        super(ImageLogger, self).__init__()
        self.val_dataset = val_dataset.take(1)
        self.log_dir = log_dir
        self.model_name = model_name.lower()
        self.freq = freq
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return

        print(f"\n[ImageLogger] Generating long preview for Epoch {epoch + 1}...")

        all_rows = []
        is_pix2pix = (self.model_name == 'pix2pix')

        for inp, targ in self.val_dataset:
            # inference
            if is_pix2pix:
                pred = self.model.generator(inp, training=False)
            else:
                pred = self.model(inp, training=False)

            # convert to Numpy
            inp_np = inp.numpy()
            pred_np = pred.numpy()
            targ_np = targ.numpy()

            # denormalize
            if is_pix2pix:
                # [-1, 1] to [0, 255]
                inp_np = (inp_np + 1.0) * 127.5
                pred_np = (pred_np + 1.0) * 127.5
                targ_np = (targ_np + 1.0) * 127.5
            else:
                # [0, 1] to [0, 255]
                inp_np = inp_np * 255.0
                pred_np = pred_np * 255.0
                targ_np = targ_np * 255.0

            # stack images horizontally: input, prediction, target
            for i in range(min(inp.shape[0], 8)):
                img_in = inp_np[i]
                img_pred = pred_np[i]
                img_tar = targ_np[i]

                # if grayscale, convert to 3 channel for visualization
                if img_in.shape[-1] == 1:
                    img_in = np.concatenate([img_in, img_in, img_in], axis=-1)
                    img_pred = np.concatenate([img_pred, img_pred, img_pred], axis=-1)
                    img_tar = np.concatenate([img_tar, img_tar, img_tar], axis=-1)

                row = np.hstack((img_in, img_pred, img_tar))
                all_rows.append(row)

        # save the full grid
        if len(all_rows) > 0:
            full_grid = np.vstack(all_rows)
            full_grid = np.clip(full_grid, 0, 255).astype(np.uint8)
            full_grid = cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(self.log_dir, f"epoch_{epoch + 1:03d}.jpg")
            cv2.imwrite(save_path, full_grid)
            print(f"[ImageLogger] Saved preview to {save_path}")


# custom callback for saving best generator in GANs
class GANCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_gen_loss', save_best_only=True):
        super(GANCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)

        if current_loss is None:
            return

        # check if we should save
        if self.save_best_only:
            if current_loss < self.best_loss:
                print("Saving generator...")
                self.best_loss = current_loss
                # save only the generator weights
                self.model.generator.save_weights(self.filepath)
        else:
            self.model.generator.save_weights(self.filepath)
