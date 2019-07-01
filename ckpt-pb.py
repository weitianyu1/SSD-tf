# coding=UTF-8
import tensorflow as tf
import shutil
import os.path
from tensorflow.python.framework import graph_util


# MODEL_DIR = "model/pb"
# MODEL_NAME = "addmodel.pb"

# if os.path.exists(MODEL_DIR): 删除目录
#     shutil.rmtree(MODEL_DIR)
#
# if not tf.gfile.Exists(MODEL_DIR): #创建目录
#     tf.gfile.MakeDirs(MODEL_DIR)

output_graph = "model/pb/add_model.pb"

#下面的过程你可以替换成CNN、RNN等你想做的训练过程，这里只是简单的一个计算公式
input_holder = tf.placeholder(tf.float32, shape=[1], name="input_holder")
W1 = tf.Variable(tf.constant(5.0, shape=[1]), name="W1")
B1 = tf.Variable(tf.constant(1.0, shape=[1]), name="B1")
_y = (input_holder * W1) + B1
# predictions = tf.greater(_y, 50, name="predictions") #比50大返回true，否则返回false
predictions = tf.add(_y, 10,name="predictions") #做一个加法运算

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print ("predictions : ", sess.run(predictions, feed_dict={input_holder: [10.0]}))
    graph_def = tf.get_default_graph().as_graph_def() #得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程

    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess,
        graph_def,
        ["predictions"] #需要保存节点的名字
    )
    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))
    print (predictions)

# for op in tf.get_default_graph().get_operations(): 打印模型节点信息
#     print (op.name)