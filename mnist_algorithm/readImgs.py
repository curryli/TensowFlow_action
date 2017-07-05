filename = os.path.join(data_dir, trainfilename)  
    with open(filename) as fid:  
        content = fid.read()  
    content = content.split('\n')  
    content = content[:-1]  
    valuequeue = tf.train.string_input_producer(content,shuffle=True)  
    value = valuequeue.dequeue()  
    dir, labels = tf.decode_csv(records=value, record_defaults=[["string"], [""]], field_delim=" ")  
    labels = tf.string_to_number(labels, tf.int32)  
    imagecontent = tf.read_file(dir)  
    image = tf.image.decode_png(imagecontent, channels=3, dtype=tf.uint8)  
    image = tf.cast(image, tf.float32)  
    #将图片统一为32*32大小的
    image = tf.image.resize_images(image,[32,32])
    image = tf.reshape(image,[result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(image, [1, 2, 0])