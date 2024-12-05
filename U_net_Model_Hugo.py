import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(input_tensor, num_filters):
    """
    Creates a convolutional block with two convolutional layers followed by ReLU activation.
    
    Parameters:
        input_tensor (tf.Tensor): Input tensor to the block.
        num_filters (int): Number of filters for the convolutional layers.
    
    Returns:
        tf.Tensor: Output tensor of the block.
    """
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)  # Normalize activations
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    """
    Creates an encoder block with a convolutional block followed by max pooling.
    
    Parameters:
        input_tensor (tf.Tensor): Input tensor to the block.
        num_filters (int): Number of filters for the convolutional layers.
    
    Returns:
        tuple: (output of the convolutional block, output of the max pooling layer)
    """
    conv = conv_block(input_tensor, num_filters)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool

def decoder_block(input_tensor, skip_tensor, num_filters):
    """
    Creates a decoder block with an upsampling layer followed by concatenation and a conv block.
    
    Parameters:
        input_tensor (tf.Tensor): Input tensor from the previous layer.
        skip_tensor (tf.Tensor): Skip connection tensor from the encoder.
        num_filters (int): Number of filters for the convolutional layers.
    
    Returns:
        tf.Tensor: Output tensor of the block.
    """
    up = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    concat = layers.concatenate([up, skip_tensor])
    conv = conv_block(concat, num_filters)
    return conv

def build_unet(input_shape, num_classes):
    """
    Builds the U-Net architecture.
    
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes for segmentation.
    
    Returns:
        tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # Bottleneck
    bottleneck = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(bottleneck, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d4)

    # Model
    model = Model(inputs, outputs, name="U-Net")
    return model