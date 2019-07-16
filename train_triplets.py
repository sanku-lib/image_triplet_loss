import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_float('learning_rate','0.01','Learning rate')
flags.DEFINE_float('momentum','0.99', 'Momentum')
flags.DEFINE_string('model', 'conv_net', 'model to run')
flags.DEFINE_string('data_src', './data_repository/geological_similarity/', 'source of training dataset')

if __name__ == "__main__":

    # Setup Dataset
    dataset = PreProcessing(FLAGS.data_src)
    model = TripletLoss()
    placeholder_shape = [None] + list(dataset.images_train.shape[1:])
    print("placeholder_shape", placeholder_shape)

    # Setup Network
    next_batch = dataset.get_triplets_batch
    anchor_input = tf.placeholder(tf.float32, placeholder_shape, name='anchor_input')
    positive_input = tf.placeholder(tf.float32, placeholder_shape, name='positive_input')
    negative_input = tf.placeholder(tf.float32, placeholder_shape, name='negative_input')

    margin = 0.5
    anchor_output = model.conv_net(anchor_input, reuse=False)
    positive_output = model.conv_net(positive_input, reuse=True)
    negative_output = model.conv_net(negative_input, reuse=True)
    loss = model.triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum, use_nesterov=True).minimize(loss,
                                                                                                             global_step=global_step)

    # Start Training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Setup Tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

        # Train iter
        for i in range(FLAGS.train_iter):
            batch_anchor, batch_positive, batch_negative = next_batch(FLAGS.batch_size)

            _, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={anchor_input: batch_anchor, positive_input: batch_positive, negative_input: batch_negative})

            writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l)

            if (i + 1) % FLAGS.step == 0:
                saver.save(sess, "model_triplet/model.ckpt")
        saver.save(sess, "model_triplet/model.ckpt")
    print('Training completed successfully.')