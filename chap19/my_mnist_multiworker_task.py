
import tempfile
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()  # at the start!
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print(f"Starting task {resolver.task_type} #{resolver.task_id}")

# extra code – Load and split the MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28],
                                dtype=tf.uint8),
        tf.keras.layers.Rescaling(scale=1 / 255),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                               padding="same", input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                               padding="same"), 
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                               padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
                  metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)

if resolver.task_id == 0:  # the chief saves the model to the right location
    model.save("my_mnist_multiworker_model", save_format="tf")
else:
    tmpdir = tempfile.mkdtemp()  # other workers save to a temporary directory
    model.save(tmpdir, save_format="tf")
    tf.io.gfile.rmtree(tmpdir)  # and we can delete this directory at the end!
