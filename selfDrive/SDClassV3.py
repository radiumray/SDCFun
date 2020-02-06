import tensorflow as tf
from collections import OrderedDict
from collections import namedtuple
import cv2
import numpy as np
import math
import imutils
import yolo_v3
import yolo_v3_tiny
import time


class CNNBaseModel(object):
    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def layerbn(inputdata, is_training, name):
        return tf.layers.batch_normalization(inputs=inputdata, training=is_training, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)

        return ret


class FCNDecoder(CNNBaseModel):
    def __init__(self, phase):
        super(FCNDecoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def decode(self, input_tensor_dict, decode_layer_list, name):
        ret = dict()

        with tf.variable_scope(name):
            # score stage 1
            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']

            score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                kernel_size=1, use_bias=False, name='score_origin')
            decode_layer_list = decode_layer_list[1:]
            for i in range(len(decode_layer_list)):
                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
                score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))
                score = fused

            deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
                                         stride=8, use_bias=False, name='deconv_final')

            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')

            ret['logits'] = score_final

        return ret


class VGG16Encoder(CNNBaseModel):
    def __init__(self, phase):
        super(VGG16Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

            return relu

    def encode(self, input_tensor, name):
        ret = OrderedDict()

        with tf.variable_scope(name):
            # conv stage 1
            conv_1_1 = self._conv_stage(input_tensor=input_tensor, k_size=3, out_dims=64, name='conv1_1')
            conv_1_2 = self._conv_stage(input_tensor=conv_1_1, k_size=3, out_dims=64, name='conv1_2')
            pool1 = self.maxpooling(inputdata=conv_1_2, kernel_size=2, stride=2, name='pool1')

            # conv stage 2
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3, out_dims=128, name='conv2_1')
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3, out_dims=128, name='conv2_2')
            pool2 = self.maxpooling(inputdata=conv_2_2, kernel_size=2, stride=2, name='pool2')

            # conv stage 3
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3, out_dims=256, name='conv3_1')
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3, out_dims=256, name='conv3_2')
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3, out_dims=256, name='conv3_3')
            pool3 = self.maxpooling(inputdata=conv_3_3, kernel_size=2, stride=2, name='pool3')
            ret['pool3'] = dict()
            ret['pool3']['data'] = pool3
            ret['pool3']['shape'] = pool3.get_shape().as_list()

            # conv stage 4
            conv_4_1 = self._conv_stage(input_tensor=pool3, k_size=3, out_dims=512, name='conv4_1')
            conv_4_2 = self._conv_stage(input_tensor=conv_4_1, k_size=3, out_dims=512, name='conv4_2')
            conv_4_3 = self._conv_stage(input_tensor=conv_4_2, k_size=3, out_dims=512, name='conv4_3')
            pool4 = self.maxpooling(inputdata=conv_4_3, kernel_size=2, stride=2, name='pool4')
            ret['pool4'] = dict()
            ret['pool4']['data'] = pool4
            ret['pool4']['shape'] = pool4.get_shape().as_list()

            # conv stage 5
            conv_5_1 = self._conv_stage(input_tensor=pool4, k_size=3, out_dims=512, name='conv5_1')
            conv_5_2 = self._conv_stage(input_tensor=conv_5_1, k_size=3, out_dims=512, name='conv5_2')
            conv_5_3 = self._conv_stage(input_tensor=conv_5_2, k_size=3, out_dims=512, name='conv5_3')
            pool5 = self.maxpooling(inputdata=conv_5_3, kernel_size=2, stride=2, name='pool5')
            ret['pool5'] = dict()
            ret['pool5']['data'] = pool5
            ret['pool5']['shape'] = pool5.get_shape().as_list()

        return ret


class LaneNet(object):
    def __init__(self, weights_path = 'models/tusimple_lanenet/tusimple_lanenet_2019-04-10-20-11-49.ckpt-9999', \
                 gpu_use=True, gpu_memory_fraction=0.5, gpu_tf_allow_growth=True):
        super(LaneNet, self).__init__()

        tf.logging.set_verbosity(tf.logging.ERROR)

        phase = tf.constant('test', tf.string)
        self._encoder = VGG16Encoder(phase=phase)
        self._decoder = FCNDecoder(phase=phase)

        self.__width = 512
        self.__height = 256

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, self.__height, self.__width, 3], name='input_tensor')
        self.binary_seg_ret = self.__inference(input_tensor=self.input_tensor, name='lanenet_model')

        # Set sess configuration
        if gpu_use:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            sess_config = tf.ConfigProto(device_count={'CPU': 8})
        sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        sess_config.gpu_options.allow_growth = gpu_tf_allow_growth
        sess_config.gpu_options.allocator_type = 'BFC'

        # sess
        self.sess = tf.InteractiveSession(config=sess_config)

        # restore
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=weights_path)

        self.VGG_MEAN = [103.939, 116.779, 123.68]

        return

    def __build_model(self, input_tensor, name):
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                              name='decode',
                                              decode_layer_list=['pool5',
                                                                 'pool4',
                                                                 'pool3'])
            return decode_ret

    def __inference(self, input_tensor, name):
        with tf.variable_scope(name):
            # Forward propagation to get logits
            inference_ret = self.__build_model(input_tensor=input_tensor, name='inference')

            # Calculate the binary partition loss function
            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)

            return binary_seg_ret

    def pre_binarize(self, src):
        # resize
        if src.shape[0] != self.__height or src.shape[1] != self.__width:
            src = cv2.resize(src, (self.__width, self.__height), interpolation=cv2.INTER_LINEAR)

        # vgg mean
        image_input = src - self.VGG_MEAN

        return image_input

    def binarize(self, src):
        """Image binarization.

        Args:
            src (int): Input image BGR.
                       numpy.ndarray, (256, 512, 3), 0~255

        Returns:
            dst (int): Output image.
                       numpy.ndarray, (256, 512), 0~1

        """
        image_input = self.pre_binarize(src)

        # run
        binary_seg_image = self.sess.run(self.binary_seg_ret, feed_dict={self.input_tensor: [image_input]})
        dst = binary_seg_image[0].astype(np.uint8)

        return dst

    def close(self):
        self.sess.close()


class YOLOV3TF():

    def __init__(self):

        self.classes = None
        self.COLORS = []
        # self.isTiny = False
        self.data_format='NCHW' # Data format: NCHW (gpu only) / NHWC
        self.size=416
        self.conf_threshold=0.5
        self.iou_threshold=0.4
        # self.gpu_memory_fraction=0.5
        self.boxes=None
        self.inputs=None
        self.yoloSession=None
        self.scrWidth=0
        self.scrHeight=0
        self.showObjIDs=[0,1,2,3,5,6,7,9,10,11,12,13]


    def letter_box_image(self, image, output_height, output_width, fill_value):
        height_ratio = float(output_height)/image.shape[0]
        width_ratio = float(output_width)/image.shape[1]
        fit_ratio = min(width_ratio, height_ratio)
        fit_height = int(image.shape[0] * fit_ratio)
        fit_width = int(image.shape[1] * fit_ratio)

        fit_image = np.asarray(cv2.resize(image,(fit_width,fit_height), interpolation=cv2.INTER_LINEAR))

        if isinstance(fill_value, int):
            fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

        to_return = np.tile(fill_value, (output_height, output_width, 1))
        pad_top = int(0.5 * (output_height - fit_height))
        pad_left = int(0.5 * (output_width - fit_width))
        to_return[pad_top:pad_top+fit_height, pad_left:+fit_width] = fit_image
        return to_return

    def load_coco_names(self, file_name):
        names = {}
        with open(file_name) as f:
            for id, name in enumerate(f):
                names[id] = name
        return names

    def detections_boxes(self, detections):
        """
        Converts center x, center y, width and height values to coordinates of top left and bottom right points.

        :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
        :return: converted detections of same shape as input
        """
        center_x, center_y, width, height, attrs = tf.split(
            detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detections = tf.concat([boxes, attrs], axis=-1, name="output_boxes")
        return detections

    def get_boxes_and_inputs(self, model, num_classes, size, data_format):
        inputs = tf.placeholder(tf.float32, [1, size, size, 3])
        with tf.variable_scope('detector'):
            detections = model(inputs, num_classes,
                            data_format=data_format)
        boxes = self.detections_boxes(detections)
        return boxes, inputs

    def _iou(self, box1, box2):
        """
        Computes Intersection over Union value for 2 bounding boxes

        :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
        :param box2: same as box1
        :return: IoU
        """
        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        # we add small epsilon of 1e-05 to avoid division by 0
        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou


    def non_max_suppression(self, predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
        """
        Applies Non-max suppression to prediction boxes.

        :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
        :param confidence_threshold: the threshold for deciding if prediction is valid
        :param iou_threshold: the threshold for deciding if two boxes overlap
        :return: dict: class -> [(box, score)]
        """
        conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
        predictions = predictions_with_boxes * conf_mask

        result = {}
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            non_zero_idxs = np.nonzero(image_pred)
            image_pred = image_pred[non_zero_idxs]
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis=-1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                cls_scores = cls_boxes[:, -1]
                cls_boxes = cls_boxes[:, :-1]

                while len(cls_boxes) > 0:
                    box = cls_boxes[0]
                    score = cls_scores[0]
                    if cls not in result:
                        result[cls] = []
                    result[cls].append((box, score))
                    cls_boxes = cls_boxes[1:]
                    cls_scores = cls_scores[1:]
                    ious = np.array([self._iou(box, x) for x in cls_boxes])
                    iou_mask = ious < iou_threshold
                    cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]

        return result



    def rayFilteredBoxs(self, predictions_with_boxes, confidence_threshold, iou_threshold=0.4):

        conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
        predictions = predictions_with_boxes * conf_mask

        result = {}
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            non_zero_idxs = np.nonzero(image_pred)
            image_pred = image_pred[non_zero_idxs]
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis=-1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                if(cls in self.showObjIDs): #filter
                    cls_mask = classes == cls
                    cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                    cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                    cls_scores = cls_boxes[:, -1]
                    cls_boxes = cls_boxes[:, :-1]
                    while len(cls_boxes) > 0:
                        box = cls_boxes[0]
                        score = cls_scores[0]
                        if cls not in result:
                            result[cls] = []
                        result[cls].append((box, score))
                        cls_boxes = cls_boxes[1:]
                        cls_scores = cls_scores[1:]
                        ious = np.array([self._iou(box, x) for x in cls_boxes])
                        iou_mask = ious < iou_threshold
                        cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                        cls_scores = cls_scores[np.nonzero(iou_mask)]

                # for i in range(len(cls_boxes)):
                #     box = cls_boxes[i]
                #     score = cls_scores[i]
                #     if cls not in result:
                #         result[cls] = []
                #     result[cls].append((box, score))
                #     # print('object:{} score:{}'.format(box, score))
        

        return result

    def letter_box_pos_to_original_pos(self, letter_pos, current_size, ori_image_size):
        """
        Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
        :param letter_pos: The current position within letterbox image including fill value area.
        :param current_size: The size of whole image including fill value area.
        :param ori_image_size: The size of image before being letter boxed.
        :return:
        """
        letter_pos = np.asarray(letter_pos, dtype=np.float)
        current_size = np.asarray(current_size, dtype=np.float)
        ori_image_size = np.asarray(ori_image_size, dtype=np.float)
        final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
        pad = 0.5 * (current_size - final_ratio * ori_image_size)
        pad = pad.astype(np.int32)
        to_return_pos = (letter_pos - pad) / final_ratio
        return to_return_pos

    def convert_to_original_size(self, box, size, original_size, is_letter_box_image):
        if is_letter_box_image:
            box = box.reshape(2, 2)
            box[0, :] = self.letter_box_pos_to_original_pos(box[0, :], size, original_size)
            box[1, :] = self.letter_box_pos_to_original_pos(box[1, :], size, original_size)
        else:
            ratio = original_size / size
            box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))

    def draw_boxes(self, boxes, img, cls_names, detection_size, is_letter_box_image):
        height=img.shape[0]
        width=img.shape[1]
        for cls, bboxs in boxes.items():

            color = self.COLORS[cls]
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size),
                                            [width, height],
                                            is_letter_box_image)

                x=box[0]
                y=box[1]
                w=box[2]
                h=box[3]

                cv2.rectangle(img, (x,y), (w,h), color, 2)
                cv2.putText(img, '{} {:.2f}%'.format(cls_names[cls], score * 100), (int(x-10),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img


    def getFinalObjBoxsAndDistance(self, deepImg, boxes, width, height, cls_names, detection_size, is_letter_box_image):
        finalBoxs={}
        for cls, bboxs in boxes.items():
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size),
                                            [width, height],
                                            is_letter_box_image)

                x0=int(box[0])
                y0=int(box[1])
                x1=int(box[2])
                y1=int(box[3])
                # The pixels with pure white means depth of 100m 
                # or more while pure black means depth of 0 meters.
                areaValues=deepImg[y0:y1, x0:x1]
                objHeight, objWidth=areaValues.shape

                if(objHeight>0 and objWidth>0):
                    if cls not in finalBoxs:
                        finalBoxs[cls] = []
                    distance=areaValues.min()
                    finalBoxs[cls].append((box, distance))
        return finalBoxs

    def getFinalObjBoxs(self, filterBoxes, width, height, cls_names, detection_size, is_letter_box_image):
        finalBoxs={}
        for cls, bboxs in filterBoxes.items():
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size),
                                            [width, height],
                                            is_letter_box_image)
                x0=int(box[0])
                y0=int(box[1])
                x1=int(box[2])
                y1=int(box[3])
                if cls not in finalBoxs:
                    finalBoxs[cls] = []
                finalBoxs[cls].append((box, score))
        return finalBoxs

    def getDeepMask(self, objBoxsDist, width, height, cls_names):
        img = np.zeros([height,width,3],dtype=np.uint8)
        for cls, bboxs in objBoxsDist.items():
            color = self.COLORS[cls]
            for box, distance in bboxs:
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)
                cv2.putText(img, '{}'.format(cls_names[cls][:-1]), (int(x0-10),int(y0-10)), cv2.FONT_ITALIC, 0.5, color, 1)
                cv2.putText(img, '{}'.format(str(distance)), (int(x1-10),int(y1-10)), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
        return img

    def getMask(self, boxes, width, height, cls_names, detection_size, is_letter_box_image):

        img = np.zeros([height,width,3],dtype=np.uint8)
        for cls, bboxs in boxes.items():

            color = self.COLORS[cls]
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size),
                                            [width, height],
                                            is_letter_box_image)
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)
                cv2.putText(img, '{}'.format(cls_names[cls][:-1]), (int(x0-10),int(y0-10)), cv2.FONT_ITALIC, 0.5, color, 1)
        return img

    def yoloGetBoxs(self, img):
        #预处理图像
        img_resized = self.letter_box_image(img, self.size, self.size, 128)
        img_resized = img_resized.astype(np.float32)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        tStart=time.time()
        #预测
        detected_boxes = self.yoloSession.run(self.boxes, feed_dict={self.inputs: [img_resized]})

        # filtered_boxes = self.non_max_suppression(detected_boxes, confidence_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
        filtered_boxes = self.rayFilteredBoxs(detected_boxes, confidence_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)

        # print("yolo found in {}s".format(time.time() - tStart))
        
        return filtered_boxes
    
    def yoloLoadModule(self, className, isTiny, gpu_memory_fraction, ckpt_file, conf_threshold, iou_threshold, data_format='NCHW'):
        self.data_format=data_format
        self.classes = self.load_coco_names(className)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.conf_threshold=conf_threshold
        self.iou_threshold=iou_threshold

        if(isTiny):
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        self.boxes, self.inputs = self.get_boxes_and_inputs(model, len(self.classes), self.size, self.data_format)


        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        sess_config = tf.ConfigProto(device_count={'GPU': 1})

        sess_config.gpu_options.allow_growth = True # fix CUDNN_STATUS_ALLOC_FAILED Error

        sess = tf.Session(config=sess_config)
        sess.as_default()
        saver.restore(sess, ckpt_file)
        self.yoloSession=sess


        '''
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
        )

        config.gpu_options.allow_growth = True # fix CUDNN_STATUS_ALLOC_FAILED Error

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        sess = tf.Session(config=config)
        saver.restore(sess, ckpt_file)
        self.yoloSession=sess
        '''

pixelPoint = namedtuple('pixelPoint', ['x', 'y'])


class LaneLine:
    def __init__(self, pointTop, pointBot):
        self.pointTop = pointTop
        self.pointBot = pointBot
        self.laneFarMiddle = 0

        self.K = 1
        if(self.pointTop.x == self.pointBot.x):
            self.K = None
        elif(self.pointTop.y==self.pointBot.y):
            # self.K = 0.000000001
            self.K =None
        else:
            self.K = (self.pointBot.y-self.pointTop.y) / \
                (self.pointBot.x-self.pointTop.x)
        if(self.K is None):
            self.B =None
        else:
            self.B = self.pointTop.y-(self.K*self.pointTop.x)

        self.dist = math.sqrt((self.pointTop.x-self.pointBot.x)
                              ** 2 + (self.pointTop.y-self.pointBot.y)**2)

    def getKB(self):
        return self.K, self.B

    def getDistance(self):
        return self.dist
    
    def getDistanceByPoint(self, x0, y0, x1, y1):
        self.dist = math.sqrt((x0-x1) ** 2 + (y0-y1)**2)
        return self.dist
    
    def getYByX(self, x):
        if(self.K is None or self.B is None):
            return None
        else:
            y=int(self.K*x+self.B)
            return y
    
    def getXByY(self, y):
        if(self.K is None or self.B is None):
            return None
        else:
            x=int((y-self.B)/self.K)
            return x
    
    def setDistance(self, dist):
        self.dist=dist
    
    def setLaneFarMiddle(self, laneFarMiddle):
        self.laneFarMiddle=laneFarMiddle
    
    def getLaneFarMiddle(self):
        return self.laneFarMiddle


class LaneContours(object):
    def __init__(self, width=512, height=256):
        self.width=width
        self.height=height
        self.roadWidth=50

        #车头x坐标
        self.middlePiont=int(self.width/2)

        self.leftLaneBase=self.middlePiont
        self.rightLaneBase=self.middlePiont
        self.predictPos=self.middlePiont

    # @staticmethod
    def isRasonableArea(self, x):
        return cv2.contourArea(x) > 250

    # @staticmethod
    def sorByY(self, x):
        return x[0][1]
    
    def morphological_process(self, image, kernel_size=5):
        """
        morphological process to fill the hole in the binary segmentation result
        :param image:
        :param kernel_size:
        :return:
        """
        if len(image.shape) == 3:
            raise ValueError('Binary segmentation result image should be a single channel image')

        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

        # close operation fille hole
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closing

    def getLaneContoursLayer(self, BinaryImg, minLaneLen, isLaneToButtom):

        # 得到图片中的所有物体轮廓pixels列表
        contours = cv2.findContours(
            BinaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 提高兼容性
        contours = imutils.grab_contours(contours)

        # 生成空白画布
        contoursLayer = np.zeros([self.height, self.width, 3], dtype=np.uint8)

        # 定义理想车道线列表
        goodLaneLines = []

        # loop over our contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            # approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            # 从上到下排序
            sortApprox = sorted(approx, key=self.sorByY, reverse=False)
            # 获取定点和底点的x,y
            topX = sortApprox[0][0][0]
            topY = sortApprox[0][0][1]
            botX = sortApprox[-1][0][0]
            botY = sortApprox[-1][0][1]

            cv2.drawContours(contoursLayer, [approx], -1, (0, 0, 255), 1)
            # cv2.drawContours(contoursLayer, [c], -1, (0, 0, 255), 1)

            # 计算直线距离
            topPoint = pixelPoint(topX, topY)
            botPoint = pixelPoint(botX, botY)

            thisLaneLine = LaneLine(topPoint, botPoint)
            if(thisLaneLine.getDistance() > minLaneLen):
                goodLaneLines.append(thisLaneLine)

        if(isLaneToButtom):
            for i in range(len(goodLaneLines)):
                # line to bottom
                k, b = goodLaneLines[i].getKB()
                topX = goodLaneLines[i].pointTop.x
                topY = goodLaneLines[i].pointTop.y
                if(k is not None):
                    try:
                        botX = int((self.height-b)/k)
                        cv2.line(contoursLayer, (topX, topY), (botX, self.height), (0, 255, 0), 5)
                    except OverflowError:
                        print('number overflow')
        else:
            for i in range(len(goodLaneLines)):
                topX = goodLaneLines[i].pointTop.x    
                topY = goodLaneLines[i].pointTop.y
                botX = goodLaneLines[i].pointBot.x    
                botY = goodLaneLines[i].pointBot.y
                cv2.line(contoursLayer, (topX, topY), (botX, botY), (0, 255, 0), 3)

        return contoursLayer, goodLaneLines


    def getLaneLinesLayerByGoodLaneLines(self, goodLaneLines):

        # convert array to image
        frame_draw = np.zeros((self.height, self.width, 3), np.uint8)

        #距离车头中点距离列表
        laneListFarMiddle=[]

        for i in range(len(goodLaneLines)):
            # lineBase=goodLaneLines[i].pointBot.x
            k, b=goodLaneLines[i].getKB()
            lineBase=int((self.height-b)/k)
            laneFarMiddle=self.middlePiont-lineBase
            laneListFarMiddle.append(laneFarMiddle)

            goodLaneLines[i].setLaneFarMiddle(laneFarMiddle)

        # print(laneListFarMiddle)

        alist=list(map(int,laneListFarMiddle))
        laneListFarMiddle=sorted(alist,key=abs, reverse=False)
        print(laneListFarMiddle)

        laneText='none'

        leftIndex=-1
        rightIndex=-1

        for i in range(len(laneListFarMiddle)):
            if(laneListFarMiddle[i]<0):
                if(rightIndex==-1):
                    rightIndex=i
            if(laneListFarMiddle[i]>0):
                if(leftIndex==-1):
                    leftIndex=i

        for i in range(len(goodLaneLines)):
            topX = goodLaneLines[i].pointTop.x
            topY = goodLaneLines[i].pointTop.y
            botX = goodLaneLines[i].pointBot.x
            botY = goodLaneLines[i].pointBot.y

            # print('topX:{}, topY:{}, botX:{}, botY:{}'.format(topX, topY, botX, botY)) 
            laneFarMiddle=goodLaneLines[i].getLaneFarMiddle()
            indexLanePos=laneListFarMiddle.index(laneFarMiddle)


            if(indexLanePos==rightIndex):
                cv2.line(frame_draw, (topX, topY), (botX, botY), (0,0,255), 3)
                laneText='R'
                cv2.putText(frame_draw, laneText, (botX-10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)

                self.rightLaneBase=botX
                # self.rightLaneBase=self.middlePiont-laneFarMiddle

            elif(indexLanePos==leftIndex):
                cv2.line(frame_draw, (topX, topY), (botX, botY), (255,0,0), 3)
                laneText='L'
                cv2.putText(frame_draw, laneText, (botX+10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)

                self.leftLaneBase=botX
                # self.leftLaneBase=self.middlePiont-laneFarMiddle

            else:
                if(laneFarMiddle>0): #left lane
                    cv2.line(frame_draw, (topX, topY), (botX, botY), (255,0,0), 3)
                    laneText='FL'
                    cv2.putText(frame_draw, laneText, (botX+10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)
                else: #right lane
                    cv2.line(frame_draw, (topX, topY), (botX, botY), (0,0,255), 3)
                    laneText='FR'
                    cv2.putText(frame_draw, laneText, (botX-20, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)


        if(leftIndex!=-1 and rightIndex!=-1):# 如果识别了左右车道线
            self.roadWidth=abs(self.rightLaneBase-self.leftLaneBase)
            self.predictPos=self.leftLaneBase+int(self.roadWidth/2)
            cv2.line(frame_draw, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        elif(leftIndex!=-1 and rightIndex==-1):# 如果只识别到了左车道线
            self.predictPos=self.leftLaneBase+int(self.roadWidth/2)
            cv2.line(frame_draw, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        elif(leftIndex==-1 and rightIndex!=-1):# 如果只识别到了右车道线
            self.predictPos=self.rightLaneBase-int(self.roadWidth/2)
            cv2.line(frame_draw, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        else:# 没有识别任何车道线
            cv2.line(frame_draw, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)

        # 画车头中心点
        cv2.line(frame_draw, (self.middlePiont, self.height), (self.middlePiont, self.height-20), (0,255,255), 2)


        return frame_draw


class BirdsEyeView():

    def __init__(self):

        self.srcPoints=[]
        self.dstPoints=[]
        self.src=None
        self.dst=None
        self.M=None
        self.Minv=None
        self.width=512
        self.height=256
        #车头x坐标
        self.middlePiont=int(self.width/2)
        self.area=0

        self.roadWidth=50

        self.leftLaneBase=self.middlePiont
        self.rightLaneBase=self.middlePiont
        self.predictPos=self.middlePiont

    def init(self, width, height, upInsideOffset, downOutsideOffset, horizonUnderHalfHeight):
        self.srcPoints=[[(width / 2) - upInsideOffset, height / 2+horizonUnderHalfHeight],
                    [((width / 6) - downOutsideOffset), height],
                    [(width * 5 / 6) + downOutsideOffset, height],
                    [(width / 2 + upInsideOffset), height / 2+horizonUnderHalfHeight]]

        # self.dstPoints=[[(width / 4), 0],
        #             [(width / 4), height],
        #             [(width * 3 / 4), height],
        #             [(width * 3 / 4), 0]]

        self.dstPoints=[[(width * 3 / 8), 0],
            [(width * 3 / 8), height],
            [(width * 5 / 8), height],
            [(width * 5 / 8), 0]]
        # self.dstPoints=[[(width * 7 / 16), 0],
        #     [(width * 7 / 16), height],
        #     [(width * 9 / 16), height],
        #     [(width * 9 / 16), 0]]
        
        self.src = np.float32(self.srcPoints)
        self.dst = np.float32(self.dstPoints)

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        self.width=width
        self.height=height

        self.middlePiont=int(self.width/2)
        self.area=((upInsideOffset*2)+(2*(downOutsideOffset+width/3)))*(height/2-horizonUnderHalfHeight)/2
    
    def getWarp(self, image, AuxiliaryLine):
        if(AuxiliaryLine):
            pts = np.int32(self.srcPoints)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0))
        
        warped = cv2.warpPerspective(image, self.M, (int(self.width), int(self.height)), flags=cv2.INTER_LINEAR)
        return warped

    def getUnwarp(self, image):
        unwarped = cv2.warpPerspective(image, self.Minv, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return unwarped

    def drawAuxiliaryLine(self, image):
        pts = np.int32(self.srcPoints)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0))
    
    def drawBirdAuxLine(self, image):
        pts = np.int32(self.dstPoints)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    

    def letter_box_pos_to_original_pos_bird(self, letter_pos, current_size, ori_image_size):
        """
        Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
        :param letter_pos: The current position within letterbox image including fill value area.
        :param current_size: The size of whole image including fill value area.
        :param ori_image_size: The size of image before being letter boxed.
        :return:
        """
        letter_pos = np.asarray(letter_pos, dtype=np.float)
        current_size = np.asarray(current_size, dtype=np.float)
        ori_image_size = np.asarray(ori_image_size, dtype=np.float)
        final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
        pad = 0.5 * (current_size - final_ratio * ori_image_size)
        pad = pad.astype(np.int32)
        to_return_pos = (letter_pos - pad) / final_ratio
        return to_return_pos


    def convert_to_original_size_bird(self, box, size, original_size, is_letter_box_image):
        if is_letter_box_image:
            box = box.reshape(2, 2)
            box[0, :] = self.letter_box_pos_to_original_pos_bird(box[0, :], size, original_size)
            box[1, :] = self.letter_box_pos_to_original_pos_bird(box[1, :], size, original_size)
        else:
            ratio = original_size / size
            box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))

    def drawObjsForBirdView(self, filterBoxs, image, size, width, height, is_letter_box_image):
        for cls, bboxs in filterBoxs.items():
            print('classID:',cls)
            objCenterList = []
            objBottomList = []
            for box, score in bboxs:
                box = self.convert_to_original_size_bird(box, np.array((size, size)),[width, height],is_letter_box_image)
                # print('box:',box)
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                # cv2.rectangle(image, (x0,y0), (x1,y1), (255,0,0), 2)
                cpX=int(x0+(x1-x0)/2)
                cpY=int(y0+(y1-y0)/2)
                # cv2.circle(image,(cpX,cpY), 10, (255,0,0), -1)
                # cv2.line(image,(x0,y1),(x1,y1),(0,0,255),2)

                objCenterList.append([cpX, cpY])
                objBottomList.append(y1)
            
            old_points = np.array(objCenterList, dtype='float32')

            old_points = np.array([old_points])
            new_points = cv2.perspectiveTransform(old_points, self.M)

            index=0
            for point in new_points[0]:
                if(point[1]>0 and point[0]>0 and point[1]<256 and point[0]<512):
                    cv2.circle(image,(point[0], point[1]),10,(255,0,0), -1)
                    cv2.line(image,(point[0], objBottomList[index]), (int(point[0]+20), objBottomList[index]),(0,0,255),2)
                    cv2.line(image,(point[0], objBottomList[index]), (int(point[0]-20), objBottomList[index]),(0,0,255),2)
                index=index+1
        
        return image


    def getBirdViewObjDistLayer(self, deepBoxs, width, height, isDistance=True):
        img = np.zeros([height,width,3],dtype=np.uint8)
        cv2.line(img,(int(width/2),height),(int(width/2),height-20),(255,0,0),5)
        birdViewLocations={}
        for cls, bboxs in deepBoxs.items():
            for box, distance in bboxs:
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                # cv2.rectangle(image, (x0,y0), (x1,y1), (255,0,0), 2)
                cpX=int(x0+(x1-x0)/2)
                cpY=int(y0+(y1-y0)/2)

                if cls not in birdViewLocations:
                    birdViewLocations[cls] = []
                
                # old_points = np.array([[cpX,cpY]], dtype='float32')
                old_points = np.array([[cpX,y1]], dtype='float32')
                old_points = np.array([old_points])
                new_points = cv2.perspectiveTransform(old_points, self.M)
                birdViewLocations[cls].append((new_points, distance))

                cx,cy=new_points[0][0]
                if(isDistance):
                    by=height-distance*4  #by deep camera
                    cv2.putText(img, '{}'.format(str(distance)), (int(cx+10),int(by)), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)
                else:
                    by=cy #by perspectiveTransform
                    cv2.putText(img, '{}'.format(str(int(abs(height-by)))), (int(cx+10),int(by)), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)
                cv2.circle(img,(cx,by), 8, (0,0,255), -1)
                

        return img

    # @staticmethod
    def sorByY(self, x):
        return x[0][1]


    def getBirdViewLaneContoursLayer(self, BinaryImg, minLaneLen, isLaneToButtom):

        # 得到图片中的所有物体轮廓pixels列表
        contours = cv2.findContours(
            BinaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 提高兼容性
        contours = imutils.grab_contours(contours)

        # 生成空白画布
        contoursLayer = np.zeros([self.height, self.width, 3], dtype=np.uint8)

        cv2.line(contoursLayer,(self.middlePiont, self.height),(self.middlePiont ,self.height-20),(255,0,0),10)

        # 定义理想车道线列表
        goodLaneLines = []

        #距离车头中点距离列表
        laneListFarMiddle=[]

        # loop over our contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            # approx = cv2.approxPolyDP(c, 0.1 * peri, True)

            cv2.drawContours(contoursLayer, [approx], -1, (255, 255, 255), 1)

            # 从上到下排序
            sortApprox = sorted(approx, key=self.sorByY, reverse=False)
            # 获取定点和底点的x,y
            topX = sortApprox[0][0][0]
            topY = sortApprox[0][0][1]
            botX = sortApprox[-1][0][0]
            botY = sortApprox[-1][0][1]


            vx, vy, cx, cy = cv2.fitLine(approx, cv2.DIST_L2, 0, 0.01, 0.01)
            x0=int(cx-vx*self.width)
            y0=int(cy-vy*self.width)
            x1=int(cx+vx*self.width)
            y1=int(cy+vy*self.width)


            point0 = pixelPoint(x0, y0)
            point1 = pixelPoint(x1, y1)

            thisLaneLine = LaneLine(point0, point1)
            # k,b=thisLaneLine.getKB()

            lineTopY=topY
            lineTopX=thisLaneLine.getXByY(lineTopY)

            lineBotY=botY
            lineBotX=thisLaneLine.getXByY(lineBotY)

            if(lineBotX is not None and lineTopX is not None):
                thisDist=thisLaneLine.getDistanceByPoint(lineTopX, lineTopY, lineBotX, lineBotY)
                thisLaneLine.setDistance(thisDist)
                if(thisDist > minLaneLen):
                    # cv2.line(contoursLayer, (lineTopX, lineTopY), (lineBotX, lineBotY), (0, 255, 0))
                    # cv2.line(contoursLayer, (x0, y0), (x1, y1), (0, 255, 0))

                    # 计算设置这条车掉线底部离中心点的距离
                    lineBase=thisLaneLine.getXByY(self.height)
                    laneFarMiddle=self.middlePiont-lineBase
                    laneListFarMiddle.append(laneFarMiddle)
                    thisLaneLine.setLaneFarMiddle(laneFarMiddle)
                    goodLaneLines.append(thisLaneLine)


        alist=list(map(int,laneListFarMiddle))
        laneListFarMiddle=sorted(alist,key=abs, reverse=False)
        print(laneListFarMiddle)

        laneText='none'

        leftIndex=-1
        rightIndex=-1

        for i in range(len(laneListFarMiddle)):
            if(laneListFarMiddle[i]<0):
                if(rightIndex==-1):
                    rightIndex=i
            if(laneListFarMiddle[i]>0):
                if(leftIndex==-1):
                    leftIndex=i

        for i in range(len(goodLaneLines)):
            topY = 0
            topX = goodLaneLines[i].getXByY(topY)
            botY = self.height
            botX = goodLaneLines[i].getXByY(botY)
            

            # print('topX:{}, topY:{}, botX:{}, botY:{}'.format(topX, topY, botX, botY)) 
            laneFarMiddle=goodLaneLines[i].getLaneFarMiddle()
            indexLanePos=laneListFarMiddle.index(laneFarMiddle)


            if(indexLanePos==rightIndex):
                cv2.line(contoursLayer, (topX, topY), (botX, botY), (0,255,255), 3)
                laneText='R'
                cv2.putText(contoursLayer, laneText, (botX-10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)

                # self.rightLaneBase=botX
                self.rightLaneBase=self.middlePiont-laneFarMiddle

            elif(indexLanePos==leftIndex):
                cv2.line(contoursLayer, (topX, topY), (botX, botY), (255,255,0), 3)
                laneText='L'
                cv2.putText(contoursLayer, laneText, (botX+10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)

                # self.leftLaneBase=botX
                self.leftLaneBase=self.middlePiont-laneFarMiddle

            else:
                if(laneFarMiddle>0): #left lane
                    cv2.line(contoursLayer, (topX, topY), (botX, botY), (127,127,0), 3)
                    laneText='FL'
                    cv2.putText(contoursLayer, laneText, (botX+10, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)
                else: #right lane
                    cv2.line(contoursLayer, (topX, topY), (botX, botY), (0,127,127), 3)
                    laneText='FR'
                    cv2.putText(contoursLayer, laneText, (botX-20, botY-10), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)


        if(leftIndex!=-1 and rightIndex!=-1):# 如果识别了左右车道线
            self.roadWidth=abs(self.rightLaneBase-self.leftLaneBase)
            self.predictPos=self.leftLaneBase+int(self.roadWidth/2)
            cv2.line(contoursLayer, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        elif(leftIndex!=-1 and rightIndex==-1):# 如果只识别到了左车道线
            self.predictPos=self.leftLaneBase+int(self.roadWidth/2)
            cv2.line(contoursLayer, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        elif(leftIndex==-1 and rightIndex!=-1):# 如果只识别到了右车道线
            self.predictPos=self.rightLaneBase-int(self.roadWidth/2)
            cv2.line(contoursLayer, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)
        else:# 没有识别任何车道线
            cv2.line(contoursLayer, (self.predictPos, self.height), (self.predictPos, self.height-20), (0,255,0), 3)

        return contoursLayer, goodLaneLines



if __name__ == "__main__":
    print('hello')
    
