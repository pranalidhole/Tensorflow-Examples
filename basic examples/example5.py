#fetched and feed dictionary
# we will compute the equation-> y = Wx + b where x and b are placeholders and W is constant
import tensorflow as tf 
W = tf.constant([10,100],name='W')
x = tf.placeholder(tf.int32,name='x')
b = tf.placeholder(tf.int32, name='b')
Wx = tf.multiply (W,x,name='Wx')
y = tf.add(Wx,b,name='y')
y_ = tf.subtract(x,b,name='y_')
with tf.Session() as sess:
    print("Result of Wx:",sess.run(Wx,feed_dict={x:[3,33]}))
    print("Result of y:", sess.run(y, feed_dict={x:[3,33], b: [1,2]}))

    print("Intermidiate result of y: ", sess.run(y, feed_dict={Wx:[2,3],b: [1,2]}))
    print("two results: ", sess.run(fetches=[y,y_], feed_dict={x:[4,2],b:[1,2]}))
writer = tf.compat.v1.summary.FileWriter('./m3_example5',sess.graph)
writer.close()