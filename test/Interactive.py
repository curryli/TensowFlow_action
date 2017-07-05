import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph. We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
# Run the 'init' op
	sess.run(init_op)
# Print the initial value of 'state'
	print(sess.run(state))
# Run the op that updates 'state' and print 'state'.
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))






import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'. Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [−2. −1.]

# Close the Session when we're done.
sess.close()