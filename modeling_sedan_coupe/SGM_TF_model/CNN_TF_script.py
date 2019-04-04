import tensorflow as tf
import numpy as np
import argparse
import os

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    
    #Define model architecture
    
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    if type(features) is dict:
        features = features['input']
        
    input_layer = tf.reshape(features, [-1, 200, 200, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 200, 200, 1]
    # Output Tensor Shape: [batch_size, 200, 200, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 200, 200, 32]
    # Output Tensor Shape: [batch_size, 100, 100, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 32]
    # Output Tensor Shape: [batch_size, 100, 100, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 64]
    # Output Tensor Shape: [batch_size, 50, 50, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 50, 50, 64]
    # Output Tensor Shape: [batch_size, 25, 25, 64]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 64]
    # Output Tensor Shape: [batch_size, 25 * 25 * 64]
    pool3_flat = tf.reshape(pool3, [-1, 25 * 25 * 64])

    # Dense Layer
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 25 * 25 * 64]
    # Output Tensor Shape: [batch_size, 256]
    dense = tf.layers.dense(inputs=pool3_flat, units=256, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    #Predict Op
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits" : logits
            
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #Define model outputs
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,1], logits=logits)
        
        logging_hook = tf.train.LoggingTensorHook(
            {"loss" : loss,
            },
            every_n_iter=10)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])
    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        #Define model outputs
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,1], logits=logits)
        
        predictions = {
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        #swap this for AUROC?
        eval_metric_ops = {
          "eval_accuracy": tf.metrics.auc(
              labels=labels, predictions=predictions["probabilities"])}

        return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
#custom input function using training_image_processor with tf.data.Dataset
def image_processor_train_input_fn(image_processor, X_train, y_train, batch_size):
    
    dataset = tf.data.Dataset.from_generator(
        generator = lambda : image_processor.flow(X_train.reshape((X_train.shape[0], 200, 200, 1)), 
                                                  y_train, batch_size, shuffle = True), 
        output_types = (np.float32, np.int32))
    
    dataset = dataset.shuffle(512).repeat()
    
    return dataset

def image_processor_eval_input_fn(X_test, y_test, batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test.astype('float32') / 255.0, 
                                                  y_test.astype('int32')))
        
    return dataset.batch(batch_size)

def serving_input_receiver_fn():
    inputs = {
        'input' : tf.placeholder(tf.float32, [None, 200, 200, 1]),
    }
    
    inputs['input'] = tf.divide(inputs['input'], 255.0)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    
    #outter S3 bucket where all the outputs are stored
    parser.add_argument('--model_dir', type=str)
    
    #local dir for model artifacts
    parser.add_argument('--local_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    #local dir for tensorboard stuff
    parser.add_argument('--local_tb_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    #input data paths
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args, _ = parser.parse_known_args()
    
    data_train = np.loadtxt('{}/train.csv'.format(args.train), delimiter = ',')
    
    X_train = data_train[:, :-2]
    y_train = data_train[:, -2:]
    
    data_test = np.loadtxt('{}/test.csv'.format(args.test), delimiter = ',')
    
    X_test = data_test[:, :-2]
    y_test = data_test[:, -2:]
    
    #Shearing, no zooming
    training_image_processor = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale= 1./255,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range=15,
        horizontal_flip=True,
        data_format = 'channels_last',
    )
    
    config = tf.estimator.RunConfig(log_step_count_steps = 10,
                                    save_summary_steps = 10
                                   )

    # Create the Estimator
    car_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=args.model_dir, config = config)
    
    
    train_spec = tf.estimator.TrainSpec(
         input_fn=lambda:image_processor_train_input_fn(
             training_image_processor, X_train, y_train, 64), 
         max_steps=6000)
    
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:image_processor_eval_input_fn(X_test, y_test, 32),
        steps = None
    )
    
    tf.estimator.train_and_evaluate(car_classifier, train_spec, eval_spec)
    
    car_classifier.export_savedmodel(args.local_model_dir, serving_input_receiver_fn,
                            strip_default_attrs=True)