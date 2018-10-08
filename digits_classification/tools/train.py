import logging
import os
import random
import sys
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import adda


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--weights')
@click.option('--weights_end')
@click.option('--ignore_label', type=int)
@click.option('--solver', default='sgd')
@click.option('--seed', type=int)
def main(dataset, split, model, output, gpu, iterations, batch_size, display,
         lr, stepsize, snapshot, weights, weights_end, ignore_label, solver,
         seed):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    dataset_name = dataset
    split_name = split
    dataset = getattr(adda.data.get_dataset(dataset), split)
    model_fn = adda.models.get_model_fn(model)
    im, label = dataset.tf_ops()
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=batch_size)
    net, layers = model_fn(im_batch)
    if ignore_label is not None:
        mask = tf.not_equal(label_batch, ignore_label)
        label_batch = tf.boolean_mask(label_batch, mask)
        net = tf.boolean_mask(net, mask)
    class_loss = tf.losses.sparse_softmax_cross_entropy(label_batch, net)
    loss = tf.losses.get_total_loss()

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)
    step = optimizer.minimize(loss)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    if weights:
        var_dict = adda.util.collect_vars(model, end=weights_end)
        logging.info('Restoring weights from {}:'.format(weights))
        for src, tgt in var_dict.items():
            logging.info('    {:30} -> {:30}'.format(src, tgt.name))
        restorer = tf.train.Saver(var_list=var_dict)
        restorer.restore(sess, weights)
        
    model_vars = adda.util.collect_vars(model)
    saver = tf.train.Saver(var_list=model_vars)
    output_dir = os.path.join('snapshot', output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    losses = deque(maxlen=10)
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()
    for i in bar:
        loss_val, _ = sess.run([loss, step])
        losses.append(loss_val)
        if i % display == 0:
            logging.info('{:20} {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                loss_val,
                                np.mean(losses)))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        if (i + 1) % snapshot == 0:
            snapshot_path = saver.save(sess, os.path.join(output_dir, output),
                                       global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
