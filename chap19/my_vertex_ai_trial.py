
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--n_neurons", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--optimizer", default="adam")
args = parser.parse_args()

import tensorflow as tf

def build_model(args):
    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))
        for _ in range(args.n_hidden):
            model.add(tf.keras.layers.Dense(args.n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        opt = tf.keras.optimizers.get(args.optimizer)
        opt.learning_rate = args.learning_rate
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        return model

# extra code – loads and splits the dataset
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# extra code – use the AIP_* environment variable and create the callbacks
import os
model_dir = os.getenv("AIP_MODEL_DIR")
tensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")
checkpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")
trial_id = os.getenv("CLOUD_ML_TRIAL_ID")
tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_log_dir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)
callbacks = [tensorboard_cb, early_stopping_cb]

model = build_model(args)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                    epochs=10, callbacks=callbacks)
model.save(model_dir, save_format="tf")  # extra code

import hypertune

hypertune = hypertune.HyperTune()
hypertune.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="accuracy",  # name of the reported metric
    metric_value=max(history.history["val_accuracy"]),  # max accuracy value
    global_step=model.optimizer.iterations.numpy(),
)
