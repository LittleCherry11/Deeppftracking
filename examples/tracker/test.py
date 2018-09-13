import caffe
import numpy as np

model_def="vgg16_fc_free.prototxt"
model_weight="VGG_ILSVRC_16_layers.caffemodel"
img_path="/data/vot2015/basketball/00000001.jpg"

net=caffe.Net(model_def,model_weight,caffe.TEST)

img=caffe.io.load_image(img_path)
w,h,c=img.shape
img=img.transpose((2,0,1))
img=(img-0.5)*255
net.blobs['data'].reshape(1,c,w,h)
net.reshape()

net.blobs['data'].data[...]=img
output=net.forward()

print output.keys()
for k,v in net.blobs.items():
  print k,v.data.shape
print output['pool5'].shape
