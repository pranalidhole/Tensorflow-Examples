import tensorflow as tf 

x = tf.constant([100,200,300],name='x')
y = tf.constant([1,2,3],name='y')
sum_x = tf.reduce_sum(x,name='sum_x') #reducesum sums up all elements of x matrix
prod_y = tf.reduce_prod(y,name='prod_y') #acts on all elements of y and finds its product
final_div = tf.div(sum_x,prod_y,name='div')
final_mean = tf.reduce_mean([sum_x,prod_y],name='mean')
sess = tf.Session()
print("x:",sess.run(x))
print("y:",sess.run(y))
print("sum_x:",sess.run(sum_x))
print("prod_y:",sess.run(prod_y))
print("div: " , sess.run(final_div))
print("final_mean:",sess.run(final_mean))
writer = tf.compat.v1.summary.FileWriter('./m2_example3',sess.graph)
writer.close()
sess.close()