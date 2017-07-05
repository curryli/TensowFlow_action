import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)


# Launch the graph and run the ops.
with tf.Session() as sess:
	result = sess.run([product])
	print(result)
 