#!/usr/bin/env python3
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

class SegmentationCRNN():
    """
    CRNN model to find chord changes. 
    Using datasets in format (n_sequences, n_frames, n_features, 1)
    With chord targets, same as a CRNN ACR models, that will be preprocess inside the model. 
    """
    def __init__(self, input_shape):
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape,padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,4),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        _, a1, a2, a3 = model.output_shape
        
        # Classifier - RNN
        model.add(tensorflow.keras.layers.Reshape((a1, a2*a3), input_shape=(a1, a2, a3)))
        model.add(tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.LSTM(96, return_sequences=True))
        )
        model.add(
            tensorflow.keras.layers.Dense(2, activation='softmax')
        )

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )


        model.summary()
        self.model = model
        print("[INFO] The CRNN model was successfully created.")


    def fit(self, data, targets, dev_data, dev_targets, epochs=50):
        # Get chord changes from chord labels
        changes_targets = SegmentationCRNN.labels2changes(targets)
        dev_changes_targets = SegmentationCRNN.labels2changes(dev_targets)

        # Train model
        self.history = self.model.fit(
            data, changes_targets, epochs=epochs,
            validation_data=(dev_data, dev_changes_targets)
        )
        print("[INFO] The CRNN model was successfully trained.")


    def predict(self, data):
        return self.model.predict(data)

    @staticmethod
    def labels2changes(targets):
        seg_targets = []

        # Find chord changes
        last_chord = 0
        for sequence in targets:
            new_seg_sequence = []
            for chord in sequence:
                if last_chord == chord:
                    new_seg_sequence.append(0)
                else:
                    new_seg_sequence.append(1)
                    last_chord = chord
            seg_targets.append(np.array(new_seg_sequence))

        return np.array(seg_targets)

    @staticmethod
    def get_change_points(target_changes, sequence_length_ms):
        change_points = []
        _, n_frames = target_changes.shape
        for i, sequence in enumerate(target_changes):
            change_points.append(
                np.squeeze(
                    i*sequence_length_ms + (sequence_length_ms / n_frames) * np.array(np.where(np.array(sequence) == 1)),
                    axis=0
                )
            )

        return np.concatenate(change_points)

    def score(self, data, targets):
        # Get chord changes from chord labels
        changes_targets = SegmentationCRNN.labels2changes(targets)
        # Evaluate model 
        _, test_acc = self.model.evaluate(data, changes_targets, verbose=2)
        return test_acc
    
    def display_training_progress(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.9, 1])
        plt.legend(loc='lower right')
        plt.show()





class EncoderDecoderSegmentation():
    """
    Encoder from spectrogram sequence, decoder to segmentation graph:
    |\|\|\|\|\|
    |/|/|/|/|/|
    """
    def __init__(self, input_shape = (100, 128, 3)):
        model = tensorflow.keras.models.Sequential()

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=64,kernel_size=(3,3),input_shape=input_shape, padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))
        model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=128,kernel_size=(3,3),padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))
        model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))
        model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))

        model.add(tensorflow.keras.layers.UpSampling2D((2,2)))
        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))

        model.add(tensorflow.keras.layers.UpSampling2D((2,2)))
        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=128,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))

        model.add(tensorflow.keras.layers.UpSampling2D((2,2)))
        model.add(tensorflow.keras.layers.convolutional.Convolution2D(filters=64,kernel_size=(3,3), padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Activation('relu'))

        model.add(tensorflow.keras.layers.convolutional.Convolution2D(3, (3, 3), padding='same'))
        model.add(tensorflow.keras.layers.Activation('tanh'))

        adam = tensorflow.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,loss='mse',metrics=['mae'])


        model.summary()
        self.model = model
        print("[INFO] The segmentation model was successfully created.")


    def fit(self, data, targets, dev_data, dev_targets, epochs):
        self.model.fit(
            data, targets, epochs=epochs,
            validation_data=(dev_data, dev_targets)
        )

    def predict(self, data):
        return self.model.predict(data)



def chord_graphical_segmentations(output_shape, targets):
    """
    Get chord sequence, maybe data features, return
    |\|\|\|\|
    |/|/|/|/|
    graph, where | denotes chord change and \/ denotes how far is the situation from the moment that the chord was changed
    """
    n_frames, n_features = output_shape
    segmented_targets = []
    for sequence in targets:
        actual_chord = 0
        start = 0
        segmented_sequence = []
        for chord_ind, chord in enumerate(sequence):
            if not actual_chord == chord:
                for i in range(start, chord_ind):
                    n_ones = (int)(n_features - (n_features)*(((i-start)/(chord_ind-start))**(1/2)))
                    n_zeros = (int)((n_features - n_ones)/2)
                    segmented_sequence.append(
                        np.concatenate((
                            np.zeros((n_zeros)),
                            np.ones((n_ones)),
                            np.zeros((int)(max(n_features-(n_ones+n_zeros), 0)))
                        ))
                    )
                start = chord_ind
                actual_chord = chord
        for i in range(start, len(sequence)):
            n_ones = (int)(n_features - (n_features)*(((i - start)/(len(sequence)-start))**(1/2)))
            n_zeros = (int)((n_features - n_ones)/2)
            segmented_sequence.append(
                np.concatenate((
                    np.zeros((n_zeros)),
                    np.ones((n_ones)),
                    np.zeros((int)(max(n_features-(n_ones+n_zeros), 0)))
                ))
            )
        segmented_targets.append(segmented_sequence)

    return segmented_targets

def colorize_spectrograms(data, channels=3, transpose=True):
    """
    Clip images and add 'channels' channels.
    """
    # transpose
    if transpose:
        transposed_data = np.array(data).swapaxes(1,2)
    else:
        transposed_data = data

    # add three channels
    colorized_data = np.repeat(transposed_data[..., np.newaxis], channels, -1)
    return colorized_data




def print_graphical_segmentation_demo():
    chords = []
    for i in range(0, 5):
        sequence = []
        for j in range(0, 500):
            if j < 50:
                sequence.append(5)
            elif j < 100:
                sequence.append(10)
            elif j < 190:
                sequence.append(23)
            elif j < 280:
                sequence.append(8)
            elif j < 350:
                sequence.append(2)
            elif j < 450:
                sequence.append(8)
            else:
                sequence.append(10)
        chords.append(sequence)

    output_shape = (500, 100)
    segmentation = np.array(chord_graphical_segmentations(output_shape, chords))
    print(segmentation.shape)

    plt.imshow(segmentation[0].T)
    plt.show()
