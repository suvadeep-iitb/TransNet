from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import random
import pickle

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import data_utils_ascad
import data_utils_aes_hd
import data_utils_aes_rd
import data_utils_dpav4
from transformer import Transformer
import evaluation_utils_ascad
import evaluation_utils_aes_hd
import evaluation_utils_aes_rd
import evaluation_utils_dpav4

import numpy as np

# GPU config
flags.DEFINE_bool("use_tpu", default=False,
      help="Use TPUs rather than plain CPUs.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_path", default="",
      help="Path to data file")
flags.DEFINE_string("dataset", default="ASCAD",
      help="Name of the dataset (ASCAD, AES_HD, AES_RD, DPAv4)")
flags.DEFINE_string("checkpoint_dir", default=None,
      help="directory for saving checkpoint.")
flags.DEFINE_bool("warm_start", default=False,
      help="Whether to warm start training from checkpoint.")
flags.DEFINE_string("result_path", default="",
      help="Path for eval results")
flags.DEFINE_bool("do_train", default=False,
      help="Whether to perform training or evaluation")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=256,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=32,
      help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
      help="number of steps for model checkpointing.")

# Model config
flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("conv_kernel_size", default=11,
      help="Kernel size of the convolution layer")
flags.DEFINE_integer("pool_size", default=1,
      help="Pooling size of the pooling layer")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")
flags.DEFINE_bool("smooth_pos_emb", default=True,
      help="use a smooth positional embedding")
flags.DEFINE_bool("untie_pos_emb", default=True,
      help="untie relative pos emb of each layer")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# Evaluation config
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off.")
flags.DEFINE_bool("output_attn", default=False,
      help="output attention probabilities")


FLAGS = flags.FLAGS


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, tr_steps, wu_steps=0, min_lr_ratio=0.0):
        self.max_lr=max_lr
        self.tr_steps=tr_steps
        self.wu_steps=wu_steps
        self.min_lr_ratio=min_lr_ratio
    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        wu_steps_float = tf.cast(self.wu_steps, tf.float32)
        tr_steps_float = tf.cast(self.tr_steps, tf.float32)
        max_lr_float =tf.cast(self.max_lr, tf.float32)
        min_lr_ratio_float = tf.cast(self.min_lr_ratio, tf.float32)

        # warmup learning rate using linear schedule
        wu_lr = (step_float/wu_steps_float) * max_lr_float

        # decay the learning rate using the cosine schedule
        global_step = tf.math.minimum(step_float-wu_steps_float, tr_steps_float-wu_steps_float)
        decay_steps = tr_steps_float-wu_steps_float
        pi = tf.constant(math.pi)
        cosine_decay = .5 * (1. + tf.math.cos(pi * global_step / decay_steps))
        decayed = (1.-min_lr_ratio_float) * cosine_decay + min_lr_ratio_float
        decay_lr = max_lr_float * decayed
        return tf.cond(step < self.wu_steps, lambda: wu_lr, lambda: decay_lr)


def create_model(n_classes):
    if FLAGS.init == "uniform":
      initializer = tf.compat.v1.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
      proj_initializer = None
    elif FLAGS.init == "normal":
      initializer = tf.compat.v1.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)

    model = Transformer(
        n_layer = FLAGS.n_layer,
        d_model = FLAGS.d_model,
        n_head = FLAGS.n_head,
        d_head = FLAGS.d_head,
        d_inner = FLAGS.d_inner,
        dropout = FLAGS.dropout,
        dropatt = FLAGS.dropatt,
        n_classes = n_classes,
        conv_kernel_size = FLAGS.conv_kernel_size,
        pool_size = FLAGS.pool_size,
        initializer = initializer,
        clamp_len = FLAGS.clamp_len,
        untie_r = FLAGS.untie_r,
        smooth_pos_emb = FLAGS.smooth_pos_emb,
        untie_pos_emb = FLAGS.untie_pos_emb,
        output_attn = FLAGS.output_attn
    )

    return model


def train(train_dataset, eval_dataset, num_train_batch, num_eval_batch, strategy, chk_name):
  # Ensure that the batch sizes are divisible by number of replicas in sync
  assert(FLAGS.train_batch_size % strategy.num_replicas_in_sync == 0)
  assert(FLAGS.eval_batch_size % strategy.num_replicas_in_sync == 0)

  ##### Create computational graph for train dataset
  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  ##### Create computational graph for eval dataset
  eval_dist_dataset = strategy.experimental_distribute_dataset(eval_dataset)

  if FLAGS.save_steps <= 0:
    FLAGS.save_steps = None
  else:
    # Set the FLAGS.save_steps to a value multiple of FLAGS.iterations
    if FLAGS.save_steps < FLAGS.iterations:
        FLAGS.save_steps = FLAGS.iterations
    else:
        FLAGS.save_steps = (FLAGS.save_steps // FLAGS.iterations) * \
                                                          FLAGS.iterations
  ##### Instantiate learning rate scheduler object
  lr_sch = LRSchedule(
          FLAGS.learning_rate, FLAGS.train_steps, \
          FLAGS.warmup_steps, FLAGS.min_lr_ratio
  )

  if FLAGS.dataset == 'AES_HD':
    nclasses = 8
  else:
    nclasses = 256

  ##### Create computational graph for model
  with strategy.scope():
    model = create_model(nclasses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    grad_norm = tf.keras.metrics.Mean('grad_norms', dtype=tf.float32)

    if FLAGS.warm_start:
      options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
      chk_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      if chk_path is None:
        tf.compat.v1.logging.info("Could not find any checkpoint, starting training from beginning")
      else:
        tf.compat.v1.logging.info("Found checkpoint: {}".format(chk_path))
        try:
          checkpoint.restore(chk_path, options=options)
          tf.compat.v1.logging.info("Restored checkpoint: {}".format(chk_path))
        except:
          tf.compat.v1.logging.info("Could not restore checkpoint, starting training from beginning")

  @tf.function
  def train_steps(iterator, steps, bsz):
    ###### Reset the states of the update variables
    train_loss.reset_states()
    grad_norm.reset_states()
    ###### The step function for one training step
    def step_fn(inps, lbls):
      lbls = tf.squeeze(lbls)
      with tf.GradientTape() as tape:
        logits = model(inps, training=True)[0]
        if FLAGS.dataset == 'AES_HD':
          per_example_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(lbls, logits),
            axis = 1
          )
        else:
          per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(lbls, logits)
        avg_loss = tf.nn.compute_average_loss(per_example_loss, \
                                            global_batch_size=bsz)
      variables = tape.watched_variables()
      gradients = tape.gradient(avg_loss, variables)
      clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)
      optimizer.apply_gradients(list(zip(clipped, variables)))
      train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
      grad_norm.update_state(gnorm)
    for _ in range(steps):
      inps, lbls = next(iterator)
      strategy.run(step_fn, args=(inps, lbls,))

  @tf.function
  def eval_steps(iterator, steps, bsz):
    ###### The step function for one evaluation step
    def step_fn(inps, lbls):
      lbls = tf.squeeze(lbls)
      logits = model(inps, training=False)[0]
      if FLAGS.dataset == 'AES_HD':
        per_example_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(lbls, logits),
          axis = 1
        )
      else:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(lbls, logits)
      avg_loss = tf.nn.compute_average_loss(per_example_loss, \
                                            global_batch_size=bsz)
      eval_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
    for _ in range(steps):
      inps, lbls = next(iterator)
      strategy.run(step_fn, args=(inps, lbls,))

  tf.compat.v1.logging.info('Starting training ... ')
  train_iter = iter(train_dist_dataset)

  cur_step = optimizer.iterations.numpy()
  while cur_step < FLAGS.train_steps:
    train_steps(train_iter, tf.convert_to_tensor(FLAGS.iterations), \
                FLAGS.train_batch_size)

    cur_step = optimizer.iterations.numpy()
    cur_loss = train_loss.result()
    gnorm = grad_norm.result()
    lr_rate = lr_sch(cur_step)

    tf.compat.v1.logging.info("[{:6d}] | gnorm {:5.2f} lr {:9.6f} "
            "| loss {:>5.2f}".format(cur_step, gnorm, lr_rate, cur_loss))

    if FLAGS.max_eval_batch <= 0:
      num_eval_iters = num_eval_batch
    else: 
      num_eval_iters = min(FLAGS.max_eval_batch, num_eval_batch)

    eval_tr_iter = iter(train_dist_dataset)
    eval_loss.reset_states()
    eval_steps(eval_tr_iter, tf.convert_to_tensor(num_eval_iters), \
               FLAGS.train_batch_size)

    cur_eval_loss = eval_loss.result()
    tf.compat.v1.logging.info("Train batches[{:5d}]                |"
                " loss {:>5.2f}".format(num_eval_iters, cur_eval_loss))

    eval_va_iter = iter(eval_dist_dataset)
    eval_loss.reset_states()
    eval_steps(eval_va_iter, tf.convert_to_tensor(num_eval_iters), \
               FLAGS.eval_batch_size)

    cur_eval_loss = eval_loss.result()
    tf.compat.v1.logging.info("Eval  batches[{:5d}]                |"
                " loss {:>5.2f}".format(num_eval_iters, cur_eval_loss))

    if FLAGS.save_steps is not None and (cur_step) % FLAGS.save_steps == 0:
      chk_path = os.path.join(FLAGS.checkpoint_dir, chk_name)
      options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
      save_path = checkpoint.save(chk_path, options=options)
      tf.compat.v1.logging.info("Model saved in path: {}".format(save_path))

  if FLAGS.save_steps is not None and (cur_step) % FLAGS.save_steps != 0:
    chk_path = os.path.join(FLAGS.checkpoint_dir, chk_name)
    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    save_path = checkpoint.save(chk_path, options=options)
    tf.compat.v1.logging.info("Model saved in path: {}".format(save_path))


def evaluate(data, strategy, chk_name):
  # Ensure that the batch size is divisible by number of replicas in sync
  assert(FLAGS.eval_batch_size % strategy.num_replicas_in_sync == 0)

  if FLAGS.dataset == 'AES_HD':
    nclasses = 8
  else:
    nclasses = 256

  ##### Create computational graph for model
  with strategy.scope():
    model = create_model(nclasses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    chk_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if chk_path is None:
      tf.compat.v1.logging.info("Could not find any checkpoint")
      return None
    tf.compat.v1.logging.info("Found checkpoint: {}".format(chk_path))
    try:
      checkpoint.read(chk_path, options=options).expect_partial()
      tf.compat.v1.logging.info("Restored checkpoint: {}".format(chk_path))
    except:
      tf.compat.v1.logging.info("Could not restore checkpoint")
      return None

  if FLAGS.output_attn:
    output = model.predict(data, steps=FLAGS.max_eval_batch)
  else:
    output = model.predict(data)
  return output


def print_hyperparams():
  tf.compat.v1.logging.info("")
  tf.compat.v1.logging.info("")
  tf.compat.v1.logging.info("use_tpu           : %s" % (FLAGS.use_tpu))
  tf.compat.v1.logging.info("data_path         : %s" % (FLAGS.data_path))
  tf.compat.v1.logging.info("dataset           : %s" % (FLAGS.dataset))
  tf.compat.v1.logging.info("checkpoint_dir    : %s" % (FLAGS.checkpoint_dir))
  tf.compat.v1.logging.info("warm_start        : %s" % (FLAGS.warm_start))
  tf.compat.v1.logging.info("result_path       : %s" % (FLAGS.result_path))
  tf.compat.v1.logging.info("do_train          : %s" % (FLAGS.do_train))
  tf.compat.v1.logging.info("learning_rate     : %s" % (FLAGS.learning_rate))
  tf.compat.v1.logging.info("clip              : %s" % (FLAGS.clip))
  tf.compat.v1.logging.info("min_lr_ratio      : %s" % (FLAGS.min_lr_ratio))
  tf.compat.v1.logging.info("warmup_steps      : %s" % (FLAGS.warmup_steps))
  tf.compat.v1.logging.info("train_batch_size  : %s" % (FLAGS.train_batch_size))
  tf.compat.v1.logging.info("eval_batch_size   : %s" % (FLAGS.eval_batch_size))
  tf.compat.v1.logging.info("train_steps       : %s" % (FLAGS.train_steps))
  tf.compat.v1.logging.info("iterations        : %s" % (FLAGS.iterations))
  tf.compat.v1.logging.info("save_steps        : %s" % (FLAGS.save_steps))
  tf.compat.v1.logging.info("n_layer           : %s" % (FLAGS.n_layer))
  tf.compat.v1.logging.info("d_model           : %s" % (FLAGS.d_model))
  tf.compat.v1.logging.info("n_head            : %s" % (FLAGS.n_head))
  tf.compat.v1.logging.info("d_head            : %s" % (FLAGS.d_head))
  tf.compat.v1.logging.info("d_inner           : %s" % (FLAGS.d_inner))
  tf.compat.v1.logging.info("dropout           : %s" % (FLAGS.dropout))
  tf.compat.v1.logging.info("dropatt           : %s" % (FLAGS.dropatt))
  tf.compat.v1.logging.info("conv_kernel_size  : %s" % (FLAGS.conv_kernel_size))
  tf.compat.v1.logging.info("pool_size         : %s" % (FLAGS.pool_size))
  tf.compat.v1.logging.info("clamp_len         : %s" % (FLAGS.clamp_len))
  tf.compat.v1.logging.info("untie_r           : %s" % (FLAGS.untie_r))
  tf.compat.v1.logging.info("smooth_pos_emb    : %s" % (FLAGS.smooth_pos_emb))
  tf.compat.v1.logging.info("untie_pos_emb     : %s" % (FLAGS.untie_pos_emb))
  tf.compat.v1.logging.info("init              : %s" % (FLAGS.init))
  tf.compat.v1.logging.info("init_std          : %s" % (FLAGS.init_std))
  tf.compat.v1.logging.info("init_range        : %s" % (FLAGS.init_range))
  tf.compat.v1.logging.info("max_eval_batch    : %s" % (FLAGS.max_eval_batch))
  tf.compat.v1.logging.info("output_attn       : %s" % (FLAGS.output_attn))
  tf.compat.v1.logging.info("")
  tf.compat.v1.logging.info("")




def main(unused_argv):
  del unused_argv  # Unused

  print_hyperparams()

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if FLAGS.dataset == 'ASCAD':
    train_data = data_utils_ascad.Dataset(data_path=FLAGS.data_path,
                         split="train")
    test_data = data_utils_ascad.Dataset(data_path=FLAGS.data_path,
                         split="test")

  elif FLAGS.dataset == 'AES_HD':
    train_data = data_utils_aes_hd.Dataset(data_path=FLAGS.data_path,
                         split="train")
    test_data = data_utils_aes_hd.Dataset(data_path=FLAGS.data_path,
                         split="test")

  elif FLAGS.dataset == 'AES_RD':
    train_data = data_utils_aes_rd.Dataset(data_path=FLAGS.data_path,
                         split="train")
    test_data = data_utils_aes_rd.Dataset(data_path=FLAGS.data_path,
                         split="test")

  elif FLAGS.dataset == 'DPAv4':
    train_data = data_utils_dpav4.Dataset(data_path=FLAGS.data_path,
                         split="train")
    test_data = data_utils_dpav4.Dataset(data_path=FLAGS.data_path,
                         split="test")
  else:
    assert False

  if FLAGS.use_tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.get_strategy()
  tf.compat.v1.logging.info("Number of accelerators: %s" % strategy.num_replicas_in_sync)

  if FLAGS.dataset == 'ASCAD':
    chk_name = 'ascad_trans'
  elif FLAGS.dataset == 'AES_HD':
    chk_name = 'aes_hd'
  elif FLAGS.dataset == 'AES_RD':
    chk_name = 'aes_rd'
  elif FLAGS.dataset == 'DPAv4':
    chk_name = 'dpav4'
  else:
    assert False

  if FLAGS.do_train:
    num_train_batch = train_data.num_samples // FLAGS.train_batch_size
    num_test_batch = test_data.num_samples // FLAGS.eval_batch_size

    tf.compat.v1.logging.info("num of train batches {}".format(num_train_batch))
    tf.compat.v1.logging.info("num of test batches {}".format(num_test_batch))

    train(train_data.GetTFRecords(FLAGS.train_batch_size, training=True), \
          test_data.GetTFRecords(FLAGS.eval_batch_size, training=True), \
          num_train_batch, num_test_batch, strategy, chk_name)
  else:
    num_test_batch = test_data.num_samples // FLAGS.eval_batch_size

    tf.compat.v1.logging.info("num of test batches {}".format(num_test_batch))

    output = evaluate(test_data.GetTFRecords(FLAGS.eval_batch_size, training=False), 
                           strategy, chk_name)
    test_scores = output[0]
    attn_outputs = output[1:]
    if test_scores is None:
      return

    if FLAGS.output_attn:
      plaintexts = test_data.plaintexts[:FLAGS.max_eval_batch*FLAGS.eval_batch_size]
      keys = test_data.keys[:FLAGS.max_eval_batch*FLAGS.eval_batch_size]
    else:
      plaintexts = test_data.plaintexts
      keys = test_data.keys

    key_rank_list = []
    for i in range(100):
      if FLAGS.dataset == 'ASCAD':
        key_ranks = evaluation_utils_ascad.compute_key_rank(test_scores, plaintexts, keys)
      elif FLAGS.dataset == 'AES_HD':
        key_ranks = evaluation_utils_aes_hd.compute_key_rank(test_scores, plaintexts, keys)
      elif FLAGS.dataset == 'AES_RD':
        key_ranks = evaluation_utils_aes_rd.compute_key_rank(test_scores, plaintexts, keys)
      elif FLAGS.dataset == 'DPAv4':
        mask = test_data.mask
        offsets = test_data.offsets
        key_ranks = evaluation_utils_dpav4.compute_key_rank(
            test_scores, plaintexts, keys, mask, offsets
        )

      key_rank_list.append(key_ranks)
    key_ranks = np.stack(key_rank_list, axis=0)

    with open(FLAGS.result_path+'.txt', 'w') as fout:
        for i in range(key_ranks.shape[0]):
            for r in key_ranks[i]:
                fout.write(str(r)+'\t')
            fout.write('\n')
        mean_ranks = np.mean(key_ranks, axis=0)
        for r in mean_ranks:
            fout.write(str(r)+'\t')
        fout.write('\n')
    tf.compat.v1.logging.info("written results in {}".format(FLAGS.result_path))

    if FLAGS.output_attn:
      pickle.dump(attn_outputs, open(FLAGS.result_path+'.pkl', 'wb'))


if __name__ == "__main__":
  tf.compat.v1.app.run()
