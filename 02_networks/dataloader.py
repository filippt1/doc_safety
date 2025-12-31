import tensorflow as tf
import os
import glob


# custom dataset loader class
class DatasetLoader:
    def __init__(self, clean_dir, degraded_dir, model_name='dncnn', grayscale=False):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.model_name = model_name.lower()
        self.file_pattern = "*.jpg"
        self.grayscale = grayscale

        if len(glob.glob(os.path.join(clean_dir, self.file_pattern))) == 0:
            print(f"WARNING: No files found in {clean_dir}")
        self.list_ds = tf.data.Dataset.list_files(os.path.join(clean_dir, self.file_pattern), shuffle=False)

    # process a single file path to load degraded and clean images
    def _process_path(self, clean_path):
        file_name = tf.strings.split(clean_path, os.sep)[-1]
        degraded_path = tf.strings.join([self.degraded_dir, os.sep, file_name])
        return self._load_image(degraded_path), self._load_image(clean_path)

    # load and preprocess an image from a given path
    def _load_image(self, path):
        c = 1 if self.grayscale else 3
        img = tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=c), tf.float32)
        # pix2pix [-1, 1], others [0, 1]
        return (img / 127.5) - 1.0 if self.model_name == 'pix2pix' else img / 255.0

    # return batched dataset
    def get_dataset(self, batch_size=16, shuffle=False):
        ds = self.list_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle: ds = ds.shuffle(1000)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
