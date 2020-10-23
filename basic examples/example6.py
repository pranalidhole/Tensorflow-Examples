#Variables
import tensorflow as tf 

W = tf.Variable([2.5,4.1],tf.float32,name='W')
x = tf.placeholder(tf.float32,name='x')
b = tf.Variable([3.2,4.5], tf.float32, name='b')

y = W * x + b
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("final result: ", sess.run(y,feed_dict={x:[6.1,4.2]}))

#updating values of variables by using assign()
number = tf.Variable(2)
multiplier = tf.Variable(3)
init = tf.global_variables_initializer()
result = number.assign(tf.multiply(number,multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print("Result ", i , " : " , sess.run(result))
        print("Incremented multiplier: ", sess.run(multiplier.assign_add(1)))