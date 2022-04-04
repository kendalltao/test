#!/usr/bin/env python3

import latticex.rosetta as rtt
import tensorflow as tf
import pandas as pd
import numpy as np
# import logging




data = pd.read_csv("demo_data.csv")

data1 = data["法院严重失信主体"].tolist()[:50000]
data2 = data["法院其他得分总计"].tolist()[:50000]
data3 = data["公安处罚得分"].tolist()[:50000]
data4 = data["公安其他得分总计"].tolist()[:50000]
data5 = data["应急证件得分"].tolist()[:50000]
data6 = data["应急其他得分总计"].tolist()[:50000]
data7 = data["税务级别得分"].tolist()[:50000]
data8 = data["税务其他得分总计"].tolist()[:50000]



rtt.activate("SecureNN")

# Get private data from every party

Input1 = rtt.private_input(0,[data1,data2,data3,data4])
Input2 = rtt.private_input(1,[data5,data6,data7,data8])




x1 = tf.placeholder(dtype=tf.float32)
x2 = tf.placeholder(dtype=tf.float32)
x3 = tf.placeholder(dtype=tf.float32)
x4 = tf.placeholder(dtype=tf.float32)
x5 = tf.placeholder(dtype=tf.float32)
x6 = tf.placeholder(dtype=tf.float32)
x7 = tf.placeholder(dtype=tf.float32)
x8 = tf.placeholder(dtype=tf.float32)



res1 = tf.less(x1,0)
res2 = tf.less(x3,0)
res3 = tf.less(x5,0)
res4 = tf.greater_equal((-50),x3)
res5 = tf.greater_equal((-30),x3) 
p2 = p1 = tf.add(tf.multiply(res1,0.5),1.0)
p1 = tf.multiply(p1,tf.add(tf.multiply(res2,0.5),1.0))
p2 = tf.multiply(p2,tf.add(tf.multiply(res3,0.1),1.0))
p2 = tf.multiply(p2,tf.add(tf.add(tf.multiply(res4,0.1),tf.multiply(res5,0.1)),1.0))
cipher_result=tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(600,x1),x2),x3),x4),tf.multiply(p1,x5)),x6),tf.multiply(p2,x7)),x8)


batch_size = 25000
batch_nums = (len(data1) + batch_size - 1) // batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Set only party a and b can get plain result
    a_and_b_can_get_plain = 0b011
    for i in range (0, batch_nums):
        sess.run(rtt.SecureReveal(cipher_result, a_and_b_can_get_plain), feed_dict = {x1: Input1[0][batch_size*i:batch_size*(i+1)], x2: Input1[1][batch_size*i:batch_size*(i+1)], 
                                                                                            x3: Input1[2][batch_size*i:batch_size*(i+1)], x4: Input1[3][batch_size*i:batch_size*(i+1)], 
                                                                                            x5: Input2[0][batch_size*i:batch_size*(i+1)], x6: Input2[1][batch_size*i:batch_size*(i+1)], 
                                                                                            x7: Input2[2][batch_size*i:batch_size*(i+1)], x8: Input2[3][batch_size*i:batch_size*(i+1)]})

rtt.deactivate()