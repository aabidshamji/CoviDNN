import sys
import yaml
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import VGG16, VGG19, ResNet50, DenseNet121, MobileNet, InceptionV3, ResNet50V2, Xception

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 200
LEARNING_RATE = 0.001
MOMENTUM = 0.9

def define_model(model_type):

    model = Sequential()

    # get models from ~/.keras/models

    if model_type == 'VGG16':
        model = VGG16(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'VGG19':
        model = VGG19(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'ResNet50':
        model = ResNet50(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'DenseNet121':
        model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'MobileNet':
        model = MobileNet(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'InceptionV3':
        model = InceptionV3(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'ResNet50V2':
        model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    elif model_type == 'Xception':
        model = Xception(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    for layer in model.layers:
        layer.trainable = False

    if model_type == 'default':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))

    # add new classifier layer
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def get_data():
    # create data generators
    train_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0/255.0,
        validation_split=0.2,
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        class_mode='binary',
        subset='training',
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))
    validate_generator = train_datagen.flow_from_directory(
        'data/train/',
        class_mode='binary',
        subset='validation',
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))
    test_generator = test_datagen.flow_from_directory(
        'data/test/',
        class_mode='binary',
        batch_size=1,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    return train_generator, validate_generator, test_generator


def summarize_diagnostics(history, filename):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    pyplot.legend()
    pyplot.savefig(f'output/graphs/{filename}_plot.png')
    pyplot.close()


def run_test_harness(model_type):

    model = define_model(model_type=model_type)

    train_generator, validate_generator, test_generator = get_data()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(train_generator),
                                  validation_data=validate_generator,
                                  validation_steps=len(validate_generator),
                                  epochs=EPOCHS)

    model.save(f'models/{model_type}.h5')

    loss, acc = model.evaluate_generator(test_generator, steps=len(test_generator))

    print('> %.3f' % (acc * 100.0))

    summarize_diagnostics(history, model_type)

    name = f'output/{model_type}.yaml'
    params = {
        'loss' : history.history['loss'],
        'val_loss' : history.history['val_loss'],
        'accuracy' : history.history['accuracy'],
        'val_accuracy' : history.history['val_accuracy'],
        'test_accuracy' : acc,
        'test_loss' : loss
    }

    with open(name, 'w+') as f:
        yaml.dump(params, f, allow_unicode=True)

    return acc

if __name__ == '__main__':
    m = sys.argv[1]
    _ = run_test_harness(m)
