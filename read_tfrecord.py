import os 
import tensorflow as tf 
from PIL import Image  

cwd='D:\\Detect_data\\train_images\\'

filename = 'D:\\Detect_data\\tfrecord\\train.tfrecords'
filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)#返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                             })#将image数据和label取出来

image = tf.decode_raw(features['img_raw'], tf.uint8)
print(image)
image = tf.reshape(image, [128, 128])  #reshape为128*128的3通道图片
#image = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
label = tf.cast(features['label'], tf.int32) #在流中抛出label张量

print(image)
print(label)

with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(23):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'P')#这里Image是之前提到的
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)


