import tensorflow as tf
import numpy as np
import pickle, os, cv2

# Configure logging and environment variables
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape


def get_num_of_classes():
    return len(os.listdir('gestures/'))


image_x, image_y = get_image_size()


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, image_x, image_y, 1], name="input")

    # Convolutional and Pooling Layers
    conv1 = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(2, 2), padding='same', activation='relu', name='conv1'
    )(input_layer)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(conv1)

    conv2 = tf.keras.layers.Conv2D(
        filters=32, kernel_size=(5, 5), padding='same', activation='relu', name='conv2'
    )(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=5, name='pool2')(conv2)

    conv3 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='conv3'
    )(pool2)

    # Dense Layers with Dropout
    flat = tf.reshape(conv3, [-1, 5 * 5 * 64], name="flat")
    dense = tf.keras.layers.Dense(128, activation='relu', name='dense')(flat)
    dropout = tf.keras.layers.Dropout(0.2)(dense, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    num_classes = get_num_of_classes()
    logits = tf.keras.layers.Dense(num_classes, name='logits')(dropout)

    # Predictions
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss Calculation
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        onehot_labels, logits, from_logits=True))

    # Training Configuration
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss, global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Evaluation Metrics
    eval_metric_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # Load data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    # Create Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="tmp/cnn_model3")

    # Training hook
    logging_hook = tf.estimator.LoggingTensorHook(
        tensors={"probabilities": "softmax_tensor"}, every_n_iter=50)

    # Train the model
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train_images}, y=train_labels,
        batch_size=500, num_epochs=10, shuffle=True)
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    # Evaluate the model
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": test_images}, y=test_labels, num_epochs=1, shuffle=False)
    test_results = classifier.evaluate(input_fn=eval_input_fn)
    print(test_results)


if __name__ == "__main__":
    main()