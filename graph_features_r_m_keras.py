from numpy.core.fromnumeric import shape
import tensorflow as tf 
import numpy as np

class GFRM(tf.keras.layers.Layer):

    def __init__(self, units=37, input_dim=32):
        super(GFRM, self).__init__()

        self.units = units
        self.input_dim = input_dim

        """self.matrix = self.add_weight(
            shape=(self.input_dim, self.input_dim), initializer="random_normal", trainable=True
        )

        self.weight = self.add_weight(
            shape=(self.units, self.input_dim, self.input_dim), initializer="random_normal",
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(self.units,), initializer="random_normal"
        )"""

    def build(self, input_shape):
        print("input shape :")
        print(input_shape)

        self.matrix = self.add_weight(
            shape=(input_shape[1], input_shape[1]), initializer="random_normal", trainable=True,
            name="matrix"
        )

        self.weight = self.add_weight(
            shape=(self.units, input_shape[1], input_shape[1]), initializer="random_normal",
            trainable=True, name="weight"
        )

        self.bias = self.add_weight(
            shape=(self.units,), initializer="random_normal"
        )

    def forward(self):

        maps = np.ones((self.input_dim, self.input_dim))
        print("ones test")
        #print(inputs)

        #np.fill_diagonal(maps, inputs)

        
        maps = tf.math.multiply(self.matrix, maps, name=None)
        

        maps = tf.nn.relu(maps)
        print(maps)


        conv = tf.math.multiply(self.weight, maps)
        print(conv)

        conv = tf.math.reduce_sum(conv, axis=[1, 2])
        print(conv)

        conv_bias = conv + self.bias 

        
        #result = tf.math.reduce_sum(conv_bias, axis=0)

        return conv_bias
    
    def call(self, inputs):

        try:
            maps = np.ones((inputs.shape[0] ,inputs.shape[1], inputs.shape[1]))
        except:
            maps = np.ones((3 ,inputs.shape[1], inputs.shape[1]))
        print("ones test")
        print(inputs)

        try:
            for i in range(inputs.shape[0]):
                np.fill_diagonal(maps[i], inputs[i])
        except:
            #np.fill_diagonal(maps[0], inputs.numpy())
            pass

        
        maps = tf.math.multiply(self.matrix, maps, name=None)

        

        maps = tf.nn.relu(maps)
        print(maps)


        conv = tf.experimental.np.outer(maps, self.weight)
        print(conv)
        #result_conv = []
        #for i in range(maps.shape[0]):
        #    result_conv.append(tf.math.multiply(self.weight, maps[i]))
        

        print("weight")
        print(conv)

        #for i , x in enumerate(result_conv):
        #    result_conv[i] = tf.math.reduce_sum(x, axis=[1, 2])
        conv = tf.math.reduce_sum(conv, axis=[1, 2])
        print(conv)

        #for i , x in enumerate(result_conv):
        #    result_conv[i] = x + self.bias
        conv_bias = conv + self.bias
        print("shape bias")
        #print(result_conv)

        
        #result = tf.convert_to_tensor(result_conv)
        #print(result)

        return  conv_bias


        
        

    
