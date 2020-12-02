
import tensorflow as tf
import cv2
import numpy as np
'''
    仿照matlab代码自己写的一个物理传播过程，由相位得到衍射强度图
'''
from LightPipes import *
wavelength = 633*nm
N = 1024
size = 10*mm
particle = 20*um
z = 3*cm
x_shift = 400*um

pixel_pitch = 10*um
N_extend = N*5
size_extend = pixel_pitch*N_extend

F_obj = Begin(size,wavelength,N)
F_obj = CircScreen(F_obj, particle)
field = F_obj.field
#%%

# phase = cv2.imread('./phase.jpg', cv2.IMREAD_GRAYSCALE)
# phase = phase / 255.
# phase *= np.pi

# lambd = 0.6328e-3
lambd = wavelength
# deltaX = 8e-3
# deltaY = 8e-3
deltaX = size/N
deltaY = size/N
k = 2 * np.pi / lambd
# d3 = 10
d3=z

# shape = phase.shape
#
# N = int(shape[0])
# shape = tf.constant([N,N])
deltaL = N * deltaX
#
# M1 = tf.complex(1., 0.)
# a1 = M1 * tf.exp(tf.complex(0., 1.) * tf.complex(np.pi, 0.) * tf.cast(phase, tf.complex64))
a1 = tf.constant(field,dtype=tf.complex64)
shape = field.shape
A1 = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(a1, name='ffs1'), name='fft1'), name='ffs2')

r1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
s1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
deltaFX = 1 / deltaL * r1
deltaFY = 1 / deltaL * s1
meshgrid = tf.meshgrid(deltaFX, deltaFY)

h3 = tf.exp(tf.complex(0., 1.)*tf.cast(k*d3*tf.sqrt(1 - tf.pow(lambd * meshgrid[0], 2) - tf.pow(lambd * meshgrid[1], 2)), tf.complex64))
multi = tf.multiply(A1, h3)
multi = np.array(multi)

t = Begin(size,lambd,N)
t.field = multi
t_inter = Interpol(t,size,N_extend)
extend = t_inter.field

# extend = np.zeros((N_extend,N_extend))
# extend[int(N_extend/2-N/2):int(N_extend/2+N/2),int(N_extend/2-N/2):int(N_extend/2+N/2)]=multi
extend = tf.constant(extend,dtype=tf.complex64)
# U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(A1, h3), name='iffs1'), name='ifft1'), name='iffs2')
U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(extend, name='iffs1'), name='ifft1'), name='iffs2')

UA10 = tf.multiply(tf.math.conj(U1), U1)
UA10 = tf.math.real(UA10)
maxnum = tf.reduce_max(UA10)
minnum = tf.reduce_min(UA10)

M5 = tf.divide((UA10 - minnum), maxnum - minnum)
M5 *= 255.

M5 = np.array(M5, dtype='uint8')

# cv2.imwrite('./my_image.png', M5)
cv2.imshow('m5', M5)
# cv2.waitKey(0)
