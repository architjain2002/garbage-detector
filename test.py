import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = './files_for_pb/retrained_graph.pb' #path to your .pb file
with tf.compat.v1.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
wts = [n for n in graph_nodes if n.op=='Const']
from tensorflow.python.framework import tensor_util

for n in wts:
    print ("Name of the node - %s" % n.name)
    print ("Value - ") 
    print (tensor_util.MakeNdarray(n.attr['value'].tensor))