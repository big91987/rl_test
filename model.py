import tensorflow.python.keras as keras

# class CNNModel(object):
#     def __init__(self):
#         pass
#     def __call__(self, *args, **kwargs):
#         # if not 'input' in kwargs.keys():
#         #     print('input need')
#         #     return
#         # input = kwargs['input']
#
#         model = keras.Sequential()
#         model.add(keras.layers.Conv2D(
#             filters=32,
#             kernel_size=8,
#             strides=(3, 3),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Conv2D(
#             filters=64,
#             kernel_size=4,
#             strides=(2, 2),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Conv2D(
#             filters=64,
#             kernel_size=8,
#             strides=(1, 1),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Flatten())

def cnn_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=(3, 3),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=8,
        strides=(1, 1),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Flatten())
    return model

def model_generator(model_name, model_path = None):
    if model_name == 'cnn_modle':
        return cnn_model()
    else:
        return None