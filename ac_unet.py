import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import h5py
import argparse

class MyAgUnet:
    def get_batch(self, ite, batchNumber, data, label, label_map, *ref):
        idx1 = ite*batchNumber
        idx2 = (ite+1)*batchNumber
        data_b=data[idx1:idx2, :, :, :, :]
        label_map_b = label_map[idx1:idx2, :, :, :]
        temp = 0
        for refer in ref:
            temp = temp + 1
            str_ref = "ref" + str(temp) + "_b = refer[idx1:idx2, :, :, :]"
            exec (str_ref)
        str_return = "rr = (data_b, label_b, label_map_b"
        for i in range(temp):
            str_return = str_return + ", ref" + str(i + 1) + "_b"
        str_return = str_return + ")"
        exec (str_return)
        return rr

    def label_to_tensor(self, label):
        label_shape = label.shape
        label_tensor = np.zeros([label_shape[0], label_shape[1], label_shape[2], label_shape[3], 4], dtype=np.float32)
        for i in range(label_shape[0]):
            for j in range(label_shape[1]):
                for k in range(label_shape[2]):
                    for l in range(label_shape[3]):
                        label_tensor[i, j, k, l, int(label[i, j, k, l])] = 1
        return label_tensor

    def weight_variable(self, shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def weight_variable_devonc(self, shape, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv3d(self, x, W):
        conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
        return conv_3d

    def deconv3d(self, x, W, output_shape,stride, padding='SAME'):
        return tf.nn.conv3d_transpose(x, W, output_shape, [1, stride, stride, stride, 1], padding=padding)

    def max_pool3d(self, x, n):
        return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')


    def crop_and_concat3d(self, x1, x2):
        return tf.concat([x1, x2], 4)

    def batchnorm3d(self, x, is_training, eps=1e-8, decay=0.9, name='BatchNorm3d'):

        from tensorflow.python.training.moving_averages import assign_moving_average
        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)


            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))

            bn = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                            variance_epsilon=eps)
        return bn

    def cross_entropy(self, y_, output_map):
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")

    def ag_unet(self, image, label_map, w, ref_num, flag, lamb, epoch_num, iter_num, *ref):
        global xs
        global ys
        temp = 0
        for ii in range(ref_num):
            temp = temp + 1
            str_dec = "global ac" + str(temp)
            exec(str_dec)

        xs = tf.placeholder(tf.float32,shape=[None, w, w, w, 1])
        ys = tf.placeholder(tf.float32, shape=[None, w, w, w, 4])
        myis_traing = tf.placeholder(dtype=tf.bool)

        temp = 0
        for ii in range(ref_num):
            temp = temp + 1
            str_ass = "ac" + str(temp) + "= tf.placeholder(tf.float32, shape = [None, " + str(w) + ", " + str(w) + ", " + str(w) + ", 4])"
            exec(str_ass)

        W_conv1_1 = self.weight_variable([3, 3, 3, 1, 64])
        bias1_1 = self.bias_variable([64])
        conv1_1 = self.conv3d(xs, W_conv1_1)+bias1_1
        bn1_1 = tf.nn.relu(self.batchnorm3d(conv1_1, is_training=myis_traing, name='layer1_1'))

        W_conv1_2 = self.weight_variable([3, 3, 3, 64, 64])
        bias1_2 = self.bias_variable([64])
        conv1_2 = self.conv3d(bn1_1, W_conv1_2)+bias1_2
        bn1_2 = tf.nn.relu(self.batchnorm3d(conv1_2, is_training=myis_traing, name='layer1_2'))

        W_conv1_3 = self.weight_variable([3, 3, 3, 64, 64])
        bias1_3 = self.bias_variable([64])
        conv1_3 = self.conv3d(bn1_2, W_conv1_3) + bias1_3
        bn1_3 = tf.nn.relu(self.batchnorm3d(conv1_3, is_training=myis_traing, name='layer1_3'))

        pool1 = self.max_pool3d(bn1_3,2)

        W_conv2_0 = self.weight_variable([1, 1, 1, 64, 128])
        bias2_0 = self.bias_variable([128])
        conv2_0 = self.conv3d(pool1, W_conv2_0)+bias2_0
        bn2_0 = tf.nn.relu(self.batchnorm3d(conv2_0, is_training=myis_traing, name='layer2_0'))

        W_conv2_1 = self.weight_variable([3, 3, 3, 128, 128])
        bias2_1 = self.bias_variable([128])
        conv2_1 = self.conv3d(bn2_0, W_conv2_1)+bias2_1
        bn2_1 = tf.nn.relu(self.batchnorm3d(conv2_1, is_training=myis_traing, name='layer2_1'))

        W_conv2_2 = self.weight_variable([3, 3, 3, 128, 128])
        bias2_2 = self.bias_variable([128])
        conv2_2 = self.conv3d(bn2_1, W_conv2_2)+bias2_2
        bn2_2 = tf.nn.relu(self.batchnorm3d(conv2_2, is_training=myis_traing, name='layer2_2'))

        W_conv2_3 = self.weight_variable([3, 3, 3, 128, 128])
        bias2_3 = self.bias_variable([128])
        conv2_3 = self.conv3d(bn2_2, W_conv2_3) + bias2_3
        bn2_3 = tf.nn.relu(self.batchnorm3d(conv2_3, is_training=myis_traing, name='layer2_3'))

        pool2 = self.max_pool3d(bn2_3, 2)

        W_conv3_0 = self.weight_variable([1, 1, 1, 128, 256])
        bias3_0 = self.bias_variable([256])
        conv3_0 = self.conv3d(pool2, W_conv3_0) + bias3_0
        bn3_0 = tf.nn.relu(self.batchnorm3d(conv3_0, is_training=myis_traing, name='layer3_0'))

        W_conv3_1 = self.weight_variable([3, 3, 3, 256, 256])
        bias3_1 = self.bias_variable([256])
        conv3_1 = self.conv3d(bn3_0, W_conv3_1)+bias3_1
        bn3_1 = tf.nn.relu(self.batchnorm3d(conv3_1, is_training=myis_traing, name='layer3_1'))

        W_conv3_2 = self.weight_variable([3, 3, 3, 256, 256])
        bias3_2 = self.bias_variable([256])
        conv3_2 = self.conv3d(bn3_1, W_conv3_2)+bias3_2
        bn3_2 = tf.nn.relu(self.batchnorm3d(conv3_2, is_training=myis_traing, name='layer3_2'))

        W_conv3_3 = self.weight_variable([3, 3, 3, 256, 256])
        bias3_3 = self.bias_variable([256])
        conv3_3 = self.conv3d(bn3_2, W_conv3_3) + bias3_3
        bn3_3 = tf.nn.relu(self.batchnorm3d(conv3_3, is_training=myis_traing, name='layer3_3'))

        pool3 = self.max_pool3d(bn3_3, 2)

        W_conv4_0 = self.weight_variable([1, 1, 1, 256, 512])
        bias4_0 = self.bias_variable([512])
        conv4_0 = self.conv3d(pool3, W_conv4_0) + bias4_0
        bn4_0 = tf.nn.relu(self.batchnorm3d(conv4_0, is_training=myis_traing, name='layer4_0'))

        W_conv4_1 = self.weight_variable([3, 3, 3, 512, 512])
        bias4_1 = self.bias_variable([512])
        conv4_1 = self.conv3d(bn4_0, W_conv4_1) + bias4_1
        bn4_1 = tf.nn.relu(self.batchnorm3d(conv4_1, is_training=myis_traing, name='layer4_1'))

        W_conv4_2 = self.weight_variable([3, 3, 3, 512, 512])
        bias4_2 = self.bias_variable([512])
        conv4_2 = self.conv3d(bn4_1, W_conv4_2) + bias4_2
        bn4_2 = tf.nn.relu(self.batchnorm3d(conv4_2, is_training=myis_traing, name='layer4_2'))

        W_conv4_3 = self.weight_variable([3, 3, 3, 512, 512])
        bias4_3 = self.bias_variable([512])
        conv4_3 = self.conv3d(bn4_2, W_conv4_3) + bias4_3
        bn4_3 = tf.nn.relu(self.batchnorm3d(conv4_3, is_training=myis_traing, name='layer4_3'))

        W_deconv5_1 = self.weight_variable([3, 3, 3, 256, 512])
        d_bias_5_1 = self.bias_variable([256])
        deconv5_1 = tf.nn.relu(self.deconv3d(bn4_3, W_deconv5_1, tf.shape(bn3_3), 2) + d_bias_5_1)

        crop5 = self.crop_and_concat3d(deconv5_1, bn3_3)

        W_conv5_0 = self.weight_variable([1, 1, 1, 512, 256])
        bias5_0 = self.bias_variable([256])
        conv5_0 = self.conv3d(crop5, W_conv5_0) + bias5_0
        bn5_0 = tf.nn.relu(self.batchnorm3d(conv5_0, is_training=myis_traing, name='layer5_0'))

        W_conv5_1 = self.weight_variable([3, 3, 3, 256, 256])
        bias5_1 = self.bias_variable([256])
        conv5_1 = self.conv3d(bn5_0, W_conv5_1) + bias5_1
        bn5_1 = tf.nn.relu(self.batchnorm3d(conv5_1, is_training=myis_traing, name='layer5_1'))

        W_conv5_2 = self.weight_variable([3, 3, 3, 256, 256])
        bias5_2 = self.bias_variable([256])
        conv5_2 = self.conv3d(bn5_1, W_conv5_2) + bias5_2
        bn5_2 = tf.nn.relu(self.batchnorm3d(conv5_2, is_training=myis_traing, name='layer5_2'))

        W_conv5_3 = self.weight_variable([3, 3, 3, 256, 256])
        bias5_3 = self.bias_variable([256])
        conv5_3 = self.conv3d(bn5_2, W_conv5_3) + bias5_3
        bn5_3 = tf.nn.relu(self.batchnorm3d(conv5_3, is_training=myis_traing, name='layer5_3'))


        W_deconv6_1 = self.weight_variable([3, 3, 3, 128, 256])
        d_bias_6_1 = self.bias_variable([128])
        deconv6_1 = tf.nn.relu(self.deconv3d(bn5_3, W_deconv6_1, tf.shape(bn2_3), 2)+d_bias_6_1)

        crop6 = self.crop_and_concat3d(deconv6_1, bn2_3)

        W_conv6_0 = self.weight_variable([1, 1, 1, 256, 128])
        bias6_0 = self.bias_variable([128])
        conv6_0 = self.conv3d(crop6, W_conv6_0) + bias6_0
        bn6_0 = tf.nn.relu(self.batchnorm3d(conv6_0, is_training=myis_traing, name='layer6_0'))

        W_conv6_1 = self.weight_variable([3, 3, 3, 128, 128])
        bias6_1 = self.bias_variable([128])
        conv6_1 = self.conv3d(bn6_0,W_conv6_1)+bias6_1
        bn6_1 = tf.nn.relu(self.batchnorm3d(conv6_1, is_training=myis_traing, name='layer6_1'))

        W_conv6_2 = self.weight_variable([3, 3, 3, 128, 128])
        bias6_2 = self.bias_variable([128])
        conv6_2 = self.conv3d(bn6_1,W_conv6_2)+bias6_2
        bn6_2 = tf.nn.relu(self.batchnorm3d(conv6_2, is_training=myis_traing, name='layer6_2'))

        W_conv6_3 = self.weight_variable([3, 3, 3, 128, 128])
        bias6_3 = self.bias_variable([128])
        conv6_3 = self.conv3d(bn6_2, W_conv6_3) + bias6_3
        bn6_3 = tf.nn.relu(self.batchnorm3d(conv6_3, is_training=myis_traing, name='layer6_3'))

        W_deconv7_1=self.weight_variable([3, 3, 3, 64, 128])
        d_bias7_1 = self.bias_variable([64])
        deconv7_1 = tf.nn.relu(self.deconv3d(bn6_3,W_deconv7_1, tf.shape(bn1_3), 2)+d_bias7_1)

        crop7=self.crop_and_concat3d(deconv7_1,bn1_3)

        W_conv7_0 = self.weight_variable([1, 1, 1, 128, 64])
        bias7_0 = self.bias_variable([64])
        conv7_0 = self.conv3d(crop7, W_conv7_0) + bias7_0
        bn7_0 = tf.nn.relu(self.batchnorm3d(conv7_0, is_training=myis_traing, name='layer7_0'))

        W_conv7_1 = self.weight_variable([3, 3, 3, 64, 64])
        bias7_1 = self.bias_variable([64])
        conv7_1 = self.conv3d(bn7_0, W_conv7_1)+bias7_1
        bn7_1 = tf.nn.relu(self.batchnorm3d(conv7_1, is_training=myis_traing, name='layer7_1'))

        W_conv7_2 = self.weight_variable([3, 3, 3, 64, 64])
        bias7_2 = self.bias_variable([64])
        conv7_2 = self.conv3d(bn7_1, W_conv7_2)+bias7_2
        bn7_2 = tf.nn.relu(self.batchnorm3d(conv7_2, is_training=myis_traing, name='layer7_2'))

        W_conv7_3 = self.weight_variable([3, 3, 3, 64, 64])
        bias7_3 = self.bias_variable([64])
        conv7_3 = self.conv3d(bn7_2, W_conv7_3) + bias7_3
        bn7_3 = tf.nn.relu(self.batchnorm3d(conv7_3, is_training=myis_traing, name='layer7_3'))

        W_conv7_4 = self.weight_variable([1, 1, 1, 64, 4])
        bias7_4 = self.bias_variable([4])
        conv7_4 = tf.nn.softmax(self.conv3d(bn7_3, W_conv7_4)+bias7_4)

        loss_t = tf.reduce_mean(self.cross_entropy(y_s, conv7_4))

        if ref_num != 0:
            lam = lamb/float(ref_num)
        str_loss = "loss = loss_t"
        temp = 0
        for refer in range(ref_num):
            temp = temp + 1
            str_loss = str_loss + " + " + str(lam) + "*tf.reduce_mean(self.cross_entropy(ac" + str(temp) + ", conv7_4))"
        exec(str_loss)

        train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)

        init = tf.global_variables_initializer()
        str_check = "checkpoint_dir = 'unet_with_ac_" + str(lamb) + "/'"
        exec(str_check)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint file found')

            for epoch in range(epoch_num):
                for i in range(iter_num):
                    str2 = "batch_xs, batch_labels, batch_ys"
                    for ii in range(ref_num):
                        str2 = str2 + ", batch_ref" + str(ii + 1)
                    str2 = str2 + " = self.get_batch(i, 10, image, label_map"
                    for ii in range(ref_num):
                        str2 = str2 + ", ref" + str(ii + 1)
                    str2 = str2 + ")"
                    exec(str2)
                    str3 = "_, loss_entropy,  = sess.run([train_step, loss], feed_dict={xs: batch_xs, ys: self.label_to_tensor(batch_ys)"
                    for ii in range(ref_num):
                        str3 = str3 + ", ac" + str(ii + 1) + ": self.label_to_tensor(batch_ref" + str(ii + 1) + ")"
                    str3 = str3 + ", myis_traing: True})"
                    exec(str3)
                    print('epoch:', epoch, 'iteration: ', i, '  Loss:', loss_entropy)
                str_5 = "saver_path = saver.save(sess, \"unet_with_ac_" + str(lamb) + "/model.ckpt\")"
                exec(str_5)
                print("Model saved in file:", saver_path)
        return


parser = argparse.ArgumentParser()
parser.add_argument('--ref_num')
parser.add_argument('--training_data_dir')
parser.add_argument('--lamb')
parser.add_argument('--w')
parser.add_argument('--epoch_num')
parser.add_argument('--iter_num')
args = parser.parse_args()


ref_num = int(args.ref_num)
train_data_dir = args.training_data_dir
lamb = float(args.lamb)
epoch_num = int(args.epoch_num)
iter_num = int(args.iter_num)
w = int(args.w)


train_dataDic = h5py.File('train_data.mat')
train_label_mapDic = h5py.File('train_label_map.mat')

train_data = train_dataDic["train_data"]
train_label_map = train_label_mapDic["train_label_map"]
train_data = np.reshape(train_data, [-1, w, w, w, 1])
train_label_map = np.reshape(train_label_map, [-1, w, w, w, 1])


for i in range(ref_num):
    str1 = "ref"+str(i+1)+"Dic = h5py.File('traindata" + str(flag) + "//train_ref"+str(i+1)+".mat')"
    print(str1)
    exec(str1)
    str2 = "train_ref" + str(i+1) + " = ref" + str(i+1) + "Dic[\"train_ref" + str(i+1) + "\"]"
    print(str2)
    exec(str2)
    str3 = "train_ref" + str(i+1) + " = np.reshape(train_ref" + str(i+1) + ", [-1, " + str(w) + ", " + str(w) + ", " + str(w) + " ,1])"
    print(str3)
    exec(str3)

mynet = MyAgUnet()
str_unet = "mynet.ag_unet(train_data, train_label_map, w, ref_num, flag, lamb, epoch_num, iter_num,"
for i in range(ref_num):
    str_unet = str_unet + ", train_ref" + str(i + 1)
str_unet = str_unet + ")"
print(str_unet)
exec(str_unet)

