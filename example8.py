import tensorflow as tf 
import matplotlib.image as mp_img
import matplotlib.pyplot as plot 
import os

filename = './rose.jpg'
image = mp_img.imread(filename)
print("image shape: ",image.shape)
print("image array: ", image)
plot.imshow(image)
plot.show()
x = tf.Variable(image,name='x')
init = tf.global_variables_initializer()
#getting a transpose of the image-we flipped the axis of the original image
with tf.Session() as sess:
    sess.run(init)
    transpose = tf.transpose(x,perm=[1,0,2])
    result = sess.run(transpose)
    print("Transpose image shape:",result.shape)
    plot.imshow(result)
    plot.show()

#transposing through inbuild api - we do not need to specify new order of the axis
with tf.Session() as sess:
    sess.run(init)
    transpose2 = tf.image.transpose(x)
    result2 = sess.run(transpose2)
    print("Transpose image shape:",result2.shape)
    plot.imshow(result2)
    plot.show()