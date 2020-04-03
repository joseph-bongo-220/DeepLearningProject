import tensorflow as tf

def generate_network(size=512, width=1):
    "Generates a tensorflow graph for the network and returns is"

    inp = tf.compat.v1.placeholder(tf.float64, [None, size, size, 1], name='input')
    labels = tf.compat.v1.placeholder(tf.int32, [None], name='labels')

    # First convolutiona block with only one 
    output = generate_convolutional_block(inp, filters=16*width, stride=2)

    # 3 "normal" convolutional blocks
    output = generate_convolutional_block(output, filters=32*width)

    # last convolutional block without pooling
    output = generate_convolutional_block(output, filters=80*width, pool=False)

    # Global average pooling
    output = tf.reduce_mean(output, axis=[1,2], name='gap')
    # output = tf.layers.flatten(output, name='flatten')

    # Dense layer for the output, with softmax activation
    logits = tf.keras.layers.Dense(
        units=2, # 2 outputs
        kernel_initializer=tf.keras.initializers.he_normal(),
        name='logits',
    )(output)

    probabilities = tf.nn.softmax(logits, name='probabilities')
    classes = tf.argmax(logits, axis=1, name='classes')

    return inp, labels, {
        'logits': logits,
        'probabilities': probabilities,
        'classes': classes,
    }
def generate_convolutional_block(inp, filters, length=2, pool=True, stride=1):
    "Generates a convolutional block, with a couple of simple options"

    output = inp

    for i in range(length):
        # convolution
        output = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
        )(output)

        # batch normalization
        output = tf.compat.v1.layers.batch_normalization(output)

        # ReLU
        output = tf.nn.relu(output)


    if pool:
        output = tf.keras.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
        )(output)

    return output
def generate_functions(inp, labels, output):

    error = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, output['logits'])

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=8e-5)
    train = optimizer.minimize(error)

    metrics = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, output['classes']),
        'AUC': tf.compat.v1.metrics.auc(labels, output['probabilities'][:,1]),
    }

    return error, train, metrics
