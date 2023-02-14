from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import model
import config


for ckpts in os.listdir(config.checkpoint_path):
    print(ckpts)
    if "model.ckpt-" in ckpts:
    	ckpt = ckpts.split(".")[0] +"."+ ckpts.split(".")[1]

checkpoint_file = config.checkpoint_path+ckpt
output_graph_name = config.output_model_path

with tf.Graph().as_default() as graph:

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')

    # with slim.arg_scope(inception_v3_arg_scope()):
    #     logits, end_points = inception_v3(images, num_classes = 3, create_aux_logits = False, is_training = False)
    #
    # variables_to_restore = slim.get_variables_to_restore()

    # MOVING_AVERAGE_DECAY = 0.997
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()        #This line is commented if EMA is turned off
    #
    # saver = tf.train.Saver(variables_to_restore)
    f_score, f_geometry = model.model(input_images, is_training=False)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    #Setup graph def
    input_graph_def = graph.as_graph_def()
    output_node_names = "feature_fusion/Conv_7/Sigmoid,feature_fusion/concat_3"

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, checkpoint_file)

        #Exporting the graph
        print ("Exporting graph...")
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(","))

        with tf.io.gfile.GFile(output_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())