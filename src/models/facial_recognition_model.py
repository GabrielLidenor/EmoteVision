import tensorflow
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class FacialRecognitionModel(tensorflow.keras.Model):
    def __init__(self, num_classes: int = 10, input_shape: tuple = (150, 150, 3)):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = ResNet50V2(
                weights = 'imagenet',
                include_top = False,
                input_shape = self.input_shape
                )

        # ResNet50V2 is a trained model, which reduces the amount
        # of time we need to train with our data.
        # we freeze the layers to prevent our data to make our model
        # to forget what they pre-trained before.
        # we should only focus on data that the model cannot read with
        # its pre-trained data
        for layer in self.base_model.layers:
            layer.trainable = False

        self.global_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.classifier = Dense(self.num_classes, activation='softmax')


    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        feature_map = self.base_model(inputs, training = False)
        vector = self.global_pool(feature_map)
        embedding = self.dense2(self.dense1(vector))
        predictions = self.classifier(embedding)

        return predictions

