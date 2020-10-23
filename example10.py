#flipping image, cropping image, converting images into one tensor
import tensorflow as tf 
from PIL import Image

original_image_list = ["./rose.jpg",
                        "./dog.jpg",
                        "./cat.jpg",
                        "./tree.jpeg",
                        "./city.jpeg"]

#Make a queue of filenames including all images specified
filename_queue = tf.train.string_input_producer(original_image_list)

#Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    #Coordinate the loading of multiple image files
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image_list = []
    for i in range(len(original_image_list)):
        _, image_file = image_reader.read(filename_queue)
        # the underscore above is because read will return a tuple where first will be the filename hence we need to ignore that
        image = tf.image.decode_jpeg(image_file)
        image = tf.image.resize_images(image,[224,224])
        image.set_shape((224,224,3))

        image = tf.image.flip_up_down(image) #flips image upside down
        image = tf.image.central_crop(image, central_fraction=0.5)
        image_array = sess.run(image)
        print(image_array.shape)
        #converts a numpy array of a kind (224,224,3) to a tensor of shape (224, 224, 3)
        image_tensor = tf.stack(image_array)
        print(image_tensor)

        image_list.append(image_tensor) 
    coord.request_stop()
    coord.join(threads)
#converts all tensors to a single tensor with a 4th dimension 
#4 images can be accessed as (0,224,224,3),(1,224,224,3),(2,224,224,3),(3,224,224,3)
    images_tensor = tf.stack(image_list)
    print(images_tensor)

    summary_writer = tf.compat.v1.summary.FileWriter('./m3_example10',graph=sess.graph)
    
    summary_str = sess.run(tf.summary.image("images",images_tensor,max_outputs=5))
    summary_writer.add_summary(summary_str)
    summary_writer.close()