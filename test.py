from graph_features_r_m_keras import GFRM
import tensorflow as tf

tf.compat.v1.disable_eager_execution()



"""model = GFRM(input_dim=2)



result = model([2,3])
print(result)"""

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(34, input_shape=(2,)))
model.add(GFRM())
#model.add(GFRM())
#model.add(GFRM())
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit([[9,9]],[9])