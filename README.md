Dependencies

The following Python libraries are required.

sudo pip install Pillow

Running is based on the steps:

create the model file
create an image file containing a handwritten number
predict the integer
1. create the model file

The easiest way is to cd to the directory where the python files are located. Then run:

python CNN_MNIST.py
 
2. create an image file

You have to create a PNG file that contains a handwritten number. The background has to be white and the number has to be black. Any paint program should be able to do this. Also the image has to be auto cropped so that there is no border around the number.

3. predict the integer
 
python predict_2.py ‘6.jpg’



获取图像的像素值我执行以下步骤。imageprepare（）函数的代码片段显示了所有步骤的代码。
（1）载入我的手写数字的图像。
（2）将图像转换为黑白（模式“L”）
（3）确定原始图像的尺寸是最大的
（4）调整图像的大小，使得最大尺寸（醚的高度及宽度）为20像素，并且以相同的比例最小化尺寸刻度。
（5）锐化图像。这会极大地强化结果。
（6）把图像粘贴在28×28像素的白色画布上。在最大的尺寸上从顶部或侧面居中图像4个像素。最大尺寸始终是20个像素和4 + 20 + 4 = 28，最小尺寸被定位在28和缩放的图像的新的大小之间差的一半。
（7）获取新的图像（画布+居中的图像）的像素值。
（8）归一化像素值到0和1之间的一个值（这也在TensorFlow MNIST教程中完成）。其中0是白色的，1是纯黑色。从步骤7得到的像素值是与之相反的，其中255是白色的，0黑色，所以数值必须反转。下述公式包括反转和规格化（255-X）* 1.0 / 255.0
https://niektemme.com/2016/02/21/tensorflow-handwriting/
http://dataunion.org/24177.html
tensorflow 的 mnist 例子使用了28*28的图片进行训练的，白底黑字。
我们用来测试的6.jpg不是28*28标准格式，不过没问题，imageprepare帮我们进行了归一化，归一化好的图片我们保存成了sample.png （28*28）

CNN_MNIST.py主要修改见
saver = tf.train.Saver()
#sess.run(tf.initialize_all_variables())  旧版
sess.run(tf.global_variables_initializer())
......
save_path = saver.save(sess, "model2.ckpt")

虽然起的名字是model2.ckpt
但是生成的可能是model2.ckpt.data-00000-of-00001  model2.ckpt.meta model2.ckpt.index checkpoint

可以将这4个文件放到一个文件夹下面，比如./trainedModel
然后在predict_2.py中加载
saver.restore(sess, tf.train.latest_checkpoint('./trainedModel'))

源代码里面把这4个文件放在了当前目录下，所以直接saver.restore(sess, tf.train.latest_checkpoint('.'))就行了

注意以上生成的4个文件可能根据版本问题会出现不同，具体情况要根据版本来定。

