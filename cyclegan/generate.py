import numpy as np
import tensorflow as tf

from networks import generator


def _transform(ckpt_dir, imgs, scope):
    checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)

    imgs = np.expand_dims(imgs[:, :, :, 0], 3)
    imgs = (imgs.astype('float64') - 127.5) / 127.5

    with tf.device('/cpu:0'):
        placeholder = tf.placeholder(imgs.dtype, imgs.shape)
        dataset = tf.data.Dataset.from_tensor_slices(placeholder)
        dataset = dataset.batch(500, drop_remainder=False)
        iterator = dataset.make_initializable_iterator()

    with tf.variable_scope(scope):
        real2sim = generator(iterator.get_next())

    session_creator = tf.train.ChiefSessionCreator(checkpoint_filename_with_path=checkpoint_path)
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        batches = []
        sess.run(iterator.initializer, feed_dict={placeholder: imgs})
        while True:
            try:
                batches.append(real2sim.eval(session=sess))
            except tf.errors.OutOfRangeError:
                break

    transformed_imgs = np.concatenate(batches)
    transformed_imgs = ((transformed_imgs * 127.5) + 127.5).astype('uint8')
    transformed_imgs = np.repeat(transformed_imgs, 3, axis=3)

    return transformed_imgs


def clean_real_images(ckpt_dir, imgs):
    return _transform(ckpt_dir, imgs, 'ModelY2X/Generator')


def noisify_simulated_images(ckpt_dir, imgs):
    return _transform(ckpt_dir, imgs, 'ModelX2Y/Generator')
