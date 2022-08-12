
import os
from utils_net import *
import glob
from pathlib import Path
from PIL import Image

path = '/Users/zhangyunping/PycharmProjects/PhysenNet/test'

f = []
for p in path if isinstance(path, list) else [path]:
    p = Path(p)
    if p.is_dir():
        f += glob.glob(str(p/'**'/"*.*"),recursive=True)

img_files = sorted([x.replace('/',os.sep) for x in f if x.split('.')[-1].lower()=='png'])

wavelength = 633*nm
nx = 256
ny = 256
deltax = 8*um
z = 22*mm
otf = generate_otf(wavelength,nx,ny,deltax,deltax,z)


prefix = 'Lambda'+str(wavelength)+'_Nx'+str(nx)+'_Ny'+str(ny)+'_deltaXY'+str(deltax)+'_z'+str(z)
output_dir = './syn_data/'+prefix
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

i = 0
for img in img_files:
    name = [x.split('.')[0] for x in img.split(os.sep) if x.split('.')[-1].lower()=='png'][0]
    img = Image.open(img).convert('L')
    img = np.array(img.resize([nx,ny]))
    img = img/np.max(img)
    phase = np.exp(img*np.pi)
    holo = np.fft.ifft2(np.fft.fft2(phase)*otf)
    holo = np.abs(holo)**2
    intensity = holo/np.max(holo)*255
    im = Image.fromarray(intensity).convert('L')
    filename = output_dir+'/'+ name
    im.save(filename+'.png')
    i+=1
    if i%100==0:
        print(i)





