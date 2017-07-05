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
 
python predict_2.py ��6.jpg��



��ȡͼ�������ֵ��ִ�����²��衣imageprepare���������Ĵ���Ƭ����ʾ�����в���Ĵ��롣
��1�������ҵ���д���ֵ�ͼ��
��2����ͼ��ת��Ϊ�ڰף�ģʽ��L����
��3��ȷ��ԭʼͼ��ĳߴ�������
��4������ͼ��Ĵ�С��ʹ�����ߴ磨�ѵĸ߶ȼ���ȣ�Ϊ20���أ���������ͬ�ı�����С���ߴ�̶ȡ�
��5����ͼ����Ἣ���ǿ�������
��6����ͼ��ճ����28��28���صİ�ɫ�����ϡ������ĳߴ��ϴӶ�����������ͼ��4�����ء����ߴ�ʼ����20�����غ�4 + 20 + 4 = 28����С�ߴ类��λ��28�����ŵ�ͼ����µĴ�С֮����һ�롣
��7����ȡ�µ�ͼ�񣨻���+���е�ͼ�񣩵�����ֵ��
��8����һ������ֵ��0��1֮���һ��ֵ����Ҳ��TensorFlow MNIST�̳�����ɣ�������0�ǰ�ɫ�ģ�1�Ǵ���ɫ���Ӳ���7�õ�������ֵ����֮�෴�ģ�����255�ǰ�ɫ�ģ�0��ɫ��������ֵ���뷴ת��������ʽ������ת�͹�񻯣�255-X��* 1.0 / 255.0
https://niektemme.com/2016/02/21/tensorflow-handwriting/
http://dataunion.org/24177.html
tensorflow �� mnist ����ʹ����28*28��ͼƬ����ѵ���ģ��׵׺��֡�
�����������Ե�6.jpg����28*28��׼��ʽ������û���⣬imageprepare�����ǽ����˹�һ������һ���õ�ͼƬ���Ǳ������sample.png ��28*28��

CNN_MNIST.py��Ҫ�޸ļ�
saver = tf.train.Saver()
#sess.run(tf.initialize_all_variables())  �ɰ�
sess.run(tf.global_variables_initializer())
......
save_path = saver.save(sess, "model2.ckpt")

��Ȼ���������model2.ckpt
�������ɵĿ�����model2.ckpt.data-00000-of-00001  model2.ckpt.meta model2.ckpt.index checkpoint

���Խ���4���ļ��ŵ�һ���ļ������棬����./trainedModel
Ȼ����predict_2.py�м���
saver.restore(sess, tf.train.latest_checkpoint('./trainedModel'))

Դ�����������4���ļ������˵�ǰĿ¼�£�����ֱ��saver.restore(sess, tf.train.latest_checkpoint('.'))������

ע���������ɵ�4���ļ����ܸ��ݰ汾�������ֲ�ͬ���������Ҫ���ݰ汾������

