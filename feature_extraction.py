import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
import numpy as np

model = Sequential()

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
# how are these used? better to have on command line than just input these variables below?
# need to change from '' to real value to run correctly?
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.Define_integer('epochs', 50, "Number of epochs we'll run")
flags.Define_integer('batch_size', 256, "Number of items per batch")

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    '''
    define your model and hyperparams here
    make sure to adjust the number of classes based on
    the dataset: 10 for cifar10, 43 for traffic
    set number of classes as number of unique y labels
    '''
    num_classes = len(np.unique(y_train))

    #want to flatten last layer of model to classify 
    #shape of 1: takes all values except the first-- doesnt matter the number of values,
    #just matters the shape of those values (if array of pixel images, etc.)
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    #weird format for Flatten (first initalize, then invoke input after that?)
    flattened = Flatten()(inp)

    #what does Dense in Keras do? Guess it is doing softmax function on the classes after we 
    # matmul and add them with flatten? 
    densified = Dense(num_classes, activation="softmax")(flattened)
    
    #create new model that starts in shape of extracted data, and runs flatten and dense on it
    model = Model(inp, densified)

    #now compile that model based on certain loss and optimization function, looking for accuracy afterwards
    #avoids repeat code that I've made in tf many times now
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print('compiled')

    #train model
    #why use flags for epochs and batch size? 
    #validation data set above
    #shuffle it
    model.fit(X_train, y_train, nb_epoch=Flags.epochs, batch_size=Flags.batch_size, validation_data=(X_val, y_val), shuffle=True)
    print('train complete')

    #keras had option for computing metrics, but not in solution code... unnecessary?
    # loss_and_metrics = model.evaluate(X_val, y_val, batch_size=32)
    # print('i made stuff')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
