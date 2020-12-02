from utils import *
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


from LightPipes import *
wavelength = 633*nm
N = 1024
pixel_pitch = 10*um
size = pixel_pitch*N # 10mm * 10mm
particle = 50*um/2

[x1,y1,z1] = [0,0,3*cm]
[x2,y2,z2] = [-3.0*mm,1.2*mm,3*cm]
[x3,y3,z3] = [0.5*mm,0.5*mm,1*cm]


f_factor1 = 2*particle**2/(wavelength*z1)
f_factor2 = 2*particle**2/(wavelength*z2)
f_factor3 = 2*particle**2/(wavelength*z3)

N_extend = N*3
size_extend = pixel_pitch*N_extend

F = Begin(size,wavelength,N)
F_obj = CircScreen(F, particle)
field = F_obj.field
#%%
lambd = wavelength
deltaX = size/N
deltaY = size/N
k = 2 * np.pi / lambd
# d3 = 10
d = z1
deltaL = N * deltaX #size
a1 = tf.constant(field, dtype=tf.complex64)
shape = field.shape
A1 = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(a1, name='ffs1'), name='fft1'), name='ffs2')

r1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
s1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])

deltaFX = 1 / deltaL * r1
deltaFY = 1 / deltaL * s1

meshgrid = tf.meshgrid(deltaFX, deltaFY)

h = tf.exp(tf.complex(0., 1.)*tf.cast(k*d*tf.sqrt(1 - tf.pow(lambd * meshgrid[0], 2) - tf.pow(lambd * meshgrid[1], 2)), tf.complex64))

U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(A1, h), name='iffs1'), name='ifft1'), name='iffs2')
# U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(extend, name='iffs1'), name='ifft1'), name='iffs2')

M = get_Intensity(U1)
M = np.array(M)
plt.imshow(M,cmap='gray');plt.axis('off');plt.title("Fresnel diffraction");plt.show()
plt.imsave("Image.png",M,cmap='gray')