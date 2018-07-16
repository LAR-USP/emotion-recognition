import os
import os.path as osp
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.contrib.session_bundle import exporter
from keras.models import load_model
import tensorflow as tf
from keras import backend as K

def h5_to_pb(model, output_path):
    K.set_learning_phase(0)
    sess = K.get_session()
    tf.identity(model.outputs, 'output_node')

    ### convert variables to constants and save
    constant_graph = graph_util.convert_variables_to_constants(
                                                    sess,
                                                    sess.graph.as_graph_def(),
                                                    ['output_node'])

    graph_io.write_graph(constant_graph,
                         '.',
                         output_path,
                         as_text=False)


    # sess = K.get_session()
    # export_path = output_path
    # export_version = 1
    # saver = tf.train.Saver(sharded=True)
    # model_exporter = exporter.Exporter(saver)
    # model_exporter.init(sess.graph.as_graph_def(), named_graph_signatures={
    #   'inputs': exporter.generic_signature({'images': model.input}),
    #   'outputs': exporter.generic_signature({'scores': model.output})})
    # model_exporter.export(export_path, tf.constant(export_version), sess)
