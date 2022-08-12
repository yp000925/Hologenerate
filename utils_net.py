
import os
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
nm = 1e-9
mm=1e-3
um=1e-6

def generate_otf(wavelength, nx, ny, deltax,deltay, distance):
    r1 = np.linspace(-nx/2,nx/2-1,nx)
    c1 = np.linspace(-ny/2,ny/2-1,ny)
    deltaFx = 1/(nx*deltax)*r1
    deltaFy = 1/(nx*deltay)*c1
    meshgrid = np.meshgrid(deltaFx,deltaFy)
    k = 2*np.pi/wavelength
    otf = np.exp(1j*k*distance*np.sqrt(1-np.power(wavelength*meshgrid[0],2)
                                -np.power(wavelength*meshgrid[1],2)))
    otf = np.fft.fftshift(otf)
    return otf

if __name__ == "__main__":
    wavelength = 633*nm
    nx = 256
    ny = 256
    deltax = 8*um
    z = 22*mm
    otf = generate_otf(wavelength,nx,ny,deltax,deltax,z)
    img = plt.imread('USAF1951.jpg',format='Grey')
    img = img[1150:(1150+nx),1820:(1820+nx)]
    img = img/np.max(img)
    phase = np.exp(img*np.pi)
    holo = np.fft.ifft2(np.fft.fft2(phase)*otf)
    holo = np.abs(holo)
    holo = holo/np.max(holo)
    inv_otf = generate_otf(wavelength,nx,ny,deltax,deltax,-z)
    reconstruct = np.fft.ifft2(np.fft.fft2(holo)*inv_otf)
    reconstruct = np.abs(reconstruct)/np.max(np.abs(reconstruct))
    plt.imshow(holo,cmap='Greys')
    plt.title('hologram intensity')
    plt.show()
    plt.imshow(reconstruct,cmap='Greys')
    plt.title('reconstruct intensity')
    plt.show()
