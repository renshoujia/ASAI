# -*- coding: utf-8 -*-
    # gabor.py
    # 2015-7-7
    # github: https://github.com/michael92ht
    #__author__ = 'huangtao'
    
# import the necessary packages
import numpy as np
import cv2 as cv
from pylab import *
import tensorflow as tf

#定义了一个4尺度6方向的Gabor变换
#并将4尺度6方向变换函数图像及指定图像变换后的图像保存在指定文件夹
#可扩展为各种方法求纹理特征值等
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.compat.v1.disable_eager_execution()
def Gabor_u4v6(image,image_save_path):
    #图像预处理
    #print image
    image=cv.imread(image,0)
    print (image)
    img=cv.merge([image,image,image])
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    print('aaa')
    print (src)
    src_tensor= tf.convert_to_tensor(src)
    print (src_tensor)
    src_f = np.array(src, dtype=np.float32)
    print('bbb')
    print (src_f)
    src_f /= 255.
    print('bnn')
    print (src_f)



    us=[7,14,21,30]  #4种尺度
    vs=[0,30,60,90,120,150]     #6个方向
    kernel_size =2
    sig = 10                  #sigma 带宽，取常数10
    gm = 3                    #gamma 空间纵横比
    ps = 0.0                    #psi 相位
    i=0


    for u in us:
        for v in vs:
            lm = u
            th = v*np.pi/180
            kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
            kernelimg = kernel/2.+0.5
            dest = cv.filter2D(src_f, cv.CV_32F,kernel)
            i+=1
            if i == 20:
                # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
                dest_numpy =  np.power(dest,2)
                # dest_numpy = cv.cvtColor(dest_numpy, cv.COLOR_BGR2GRAY)
                # dest_numpy=repmat(dest_numpy,[1,1,3])
                img_tensor= tf.convert_to_tensor(dest_numpy)
                # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
                cv.imwrite(image_save_path + str(i) + '.tif', np.power(dest,2))
                
                # img_tensor = tf.reshape(img_tensor, [128, 64]) 
                # print("out2=",type(img_tensor))
                # dest_new = dest
            
            # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
            # cv.imwrite(image_save_path + str(i) + 'Mag.jpg', np.power(dest,2))
                # i+=1
    tf.compat.v1.disable_eager_execution()
    sess=tf.compat.v1.Session()
    print('f')
    sess.run(tf.compat.v1.global_variables_initializer())
    print('g')
    #转化为numpy数组 
    print (img_tensor)
    img_numpy=img_tensor.eval(session=sess) 
    print("out2=",type(img_numpy))
    print('a')
    print  (img_numpy)
    # #转化为tensor 
    img_tensor= tf.convert_to_tensor(img_numpy) 
    print("out2=",type(img_tensor))
    print('b')
    print (img_tensor)     

    print('*****************************')
    a =   img_tensor[0]
    aa=a.eval(session=sess) 
    print("out2=",type(aa))
    print(aa)
    print('ab')
    print  (a )



if __name__ == '__main__':
    image_save_path=r'******'
    image=r'*********'
    Gabor_u4v6(image,image_save_path)

