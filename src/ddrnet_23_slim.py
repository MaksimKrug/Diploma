import tensorflow as tf
from tensorflow.keras import layers, models
import segmentation_models as sm
from shared import get_loss

# utils
BN_MOM = 0.1
BOTTLENECK_EXPANSION = 2
BASICBLOCK_EXPANSION = 1


def conv3x3(out_planes, stride=1):
    """
    Return Conv3x3 with custom number of filters and stride
    """
    return layers.Conv2D(
        kernel_size=(3, 3),
        filters=out_planes,
        strides=stride,
        padding="same",
        use_bias=False,
    )


def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    """
    Residual basic block
    """
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes,)(x)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x


def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    """
    Bottleneck block
    """
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)(x_in)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters=planes,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters=planes * BOTTLENECK_EXPANSION, kernel_size=(1, 1), use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)

    if downsample is not None:
        residual = downsample

    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x


def DAPPPM(x_in, branch_planes, outplanes):
    """
    Deep agregation pyramid pooling module
    """
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width = [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width = [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization(momentum=BN_MOM)(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False,)(scale0)
    x_list.append(scale0)

    for i in range(len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(
            pool_size=(kernal_sizes_height[i], kernal_sizes_width[i]),
            strides=(stride_sizes_height[i], stride_sizes_width[i]),
            padding="same",
        )(x_in)
        temp = layers.BatchNormalization(momentum=BN_MOM)(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False,)(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height, width),)
        # add current and previous layer output
        temp = layers.Add()([temp, x_list[i]])
        temp = layers.BatchNormalization(momentum=BN_MOM)(temp)
        temp = layers.Activation("relu")(temp)
        # at the end apply 3*3 conv
        temp = layers.Conv2D(
            branch_planes, kernel_size=(3, 3), use_bias=False, padding="same"
        )(temp)
        # y[i+1]
        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    combined = layers.BatchNormalization(momentum=BN_MOM)(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False,)(combined)

    shortcut = layers.BatchNormalization(momentum=BN_MOM)(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False,)(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final


def segmentation_head(x_in, interplanes, outplanes, scale_factor=None):
    x = layers.BatchNormalization(momentum=BN_MOM)(x_in)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(interplanes, kernel_size=(3, 3), use_bias=False, padding="same")(
        x
    )

    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=range, padding="valid")(x)

    if scale_factor is not None:
        input_shape = tf.keras.backend.int_shape(x)
        height2 = input_shape[1] * scale_factor
        width2 = input_shape[2] * scale_factor
        x = tf.image.resize(x, size=(height2, width2))

    return x


def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D(
            ((planes * expansion)), kernel_size=(1, 1), strides=stride, use_bias=False
        )(x_in)
        downsample = layers.BatchNormalization(momentum=BN_MOM)(downsample)
        downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x


def ddrnet_23_slim(
    input_shape=[256, 256, 3],
    layers_arg=[2, 2, 2, 2],
    num_classes=11,
    planes=32,
    spp_planes=128,
    head_planes=64,
    scale_factor=8,
    augment=False,
):

    x_in = layers.Input(input_shape)

    highres_planes = planes * 2
    input_shape = tf.keras.backend.int_shape(x_in)
    height_output = input_shape[1] // 8
    width_output = input_shape[2] // 8

    layers_inside = []

    # 1 -> 1/2 first conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding="same")(x_in)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)
    # 1/2 -> 1/4 second conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=BN_MOM)(x)
    x = layers.Activation("relu")(x)

    # layer 1
    # 1/4 -> 1/4 first basic residual block not mentioned in the image
    x = make_layer(
        x, basic_block, planes, planes, layers_arg[0], expansion=BASICBLOCK_EXPANSION
    )
    layers_inside.append(x)

    # layer 2
    # 2 High :: 1/4 -> 1/8 storing results at index:1
    x = layers.Activation("relu")(x)
    x = make_layer(
        x,
        basic_block,
        planes,
        planes * 2,
        layers_arg[1],
        stride=2,
        expansion=BASICBLOCK_EXPANSION,
    )
    layers_inside.append(x)

    """
    For next layers 
    x:  low branch
    x_: high branch
    """

    # layer 3
    # 3 Low :: 1/8 -> 1/16 storing results at index:2
    x = layers.Activation("relu")(x)
    x = make_layer(
        x,
        basic_block,
        planes * 2,
        planes * 4,
        layers_arg[2],
        stride=2,
        expansion=BASICBLOCK_EXPANSION,
    )
    layers_inside.append(x)
    # 3 High :: 1/8 -> 1/8 retrieving from index:1
    x_ = layers.Activation("relu")(layers_inside[1])
    x_ = make_layer(
        x_, basic_block, planes * 2, highres_planes, 2, expansion=BASICBLOCK_EXPANSION
    )

    # Fusion 1
    # x -> 1/16 to 1/8, x_ -> 1/8 to 1/16
    # High to Low
    x_temp = layers.Activation("relu")(x_)
    x_temp = layers.Conv2D(
        planes * 4, kernel_size=(3, 3), strides=2, padding="same", use_bias=False
    )(x_temp)
    x_temp = layers.BatchNormalization(momentum=BN_MOM)(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[2])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1, 1), use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization(momentum=BN_MOM)(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output))  # 1/16 -> 1/8
    x_ = layers.Add()([x_, x_temp])  # next high branch input, 1/8

    if augment:
        temp_output = x_  # Auxiliary loss from high branch

    # layer 4
    # 4 Low :: 1/16 -> 1/32 storing results at index:3
    x = layers.Activation("relu")(x)
    x = make_layer(
        x,
        basic_block,
        planes * 4,
        planes * 8,
        layers_arg[3],
        stride=2,
        expansion=BASICBLOCK_EXPANSION,
    )
    layers_inside.append(x)
    # 4 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(
        x_,
        basic_block,
        highres_planes,
        highres_planes,
        2,
        expansion=BASICBLOCK_EXPANSION,
    )

    # Fusion 2 :: x_ -> 1/32 to 1/8, x -> 1/8 to 1/32 using two conv's
    # High to low
    x_temp = layers.Activation("relu")(x_)
    x_temp = layers.Conv2D(
        planes * 4, kernel_size=(3, 3), strides=2, padding="same", use_bias=False
    )(x_temp)
    x_temp = layers.BatchNormalization(momentum=BN_MOM)(x_temp)
    x_temp = layers.Activation("relu")(x_temp)
    x_temp = layers.Conv2D(
        planes * 8, kernel_size=(3, 3), strides=2, padding="same", use_bias=False
    )(x_temp)
    x_temp = layers.BatchNormalization(momentum=BN_MOM)(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[3])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1, 1), use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization(momentum=BN_MOM)(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output))
    x_ = layers.Add()([x_, x_temp])

    # layer 5
    # 5 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(
        x_,
        bottleneck_block,
        highres_planes,
        highres_planes,
        1,
        expansion=BOTTLENECK_EXPANSION,
    )
    x = layers.Activation("relu")(x)
    # 5 Low :: 1/32 -> 1/64
    x = make_layer(
        x,
        bottleneck_block,
        planes * 8,
        planes * 8,
        1,
        stride=2,
        expansion=BOTTLENECK_EXPANSION,
    )

    # Deep Aggregation Pyramid Pooling Module
    x = DAPPPM(x, spp_planes, planes * 4)

    # resize from 1/64 to 1/8
    x = tf.image.resize(x, (height_output, width_output))

    x_ = layers.Add()([x, x_])

    x_ = segmentation_head((x_), head_planes, num_classes, scale_factor)

    # apply softmax at the output layer
    x_ = tf.nn.softmax(x_)

    if augment:
        x_extra = segmentation_head(
            temp_output, head_planes, num_classes, scale_factor
        )  # without scaling
        x_extra = tf.nn.softmax(x_extra)
        model_output = [x_, x_extra]
    else:
        model_output = x_

    model = models.Model(inputs=[x_in], outputs=[model_output])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, "depthwise_initializer"):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model

def get_ddrnet_23_slim_model():
    # get loss
    loss = get_loss()
    # init model
    model = ddrnet_23_slim(num_classes=11, input_shape=(256, 256, 3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # compile model
    model.compile(loss=loss, optimizer=optimizer,
                    metrics=[sm.metrics.IOUScore(threshold=0.5),
                            sm.metrics.FScore(threshold=0.5)], )

    return model
