import caffe
import numpy as np
import os 
import matplotlib.pyplot as plt

root_path="/home/ccjiang/Documents/py-faster-rcnn/caffe-fast-rcnn/examples/tracker/"
model_def=os.path.join(root_path, "vgg16_align.prototxt")
model_weight=os.path.join(root_path,"VGG_ILSVRC_16_layers.caffemodel")

caffe.set_mode_gpu()
caffe.set_device(1)
net=caffe.Net(model_def, model_weight, caffe.TEST)

print net.params.keys()
print net.blobs.keys()


dataset_path="/data/OTB50"
sequence="Basketball"
id=1

frame_name="img/%04d.jpg"%id
frame_path=os.path.join(dataset_path,sequence,frame_name)

gt_path=os.path.join(dataset_path,sequence,"groundtruth_rect.txt")

try:
  gt_str=np.loadtxt(gt_path, dtype=str, delimiter='\n')
  gt_boxes=map(lambda x : map(float, x.split(',')), gt_str)
  gt_boxes=np.array(gt_boxes, dtype=float)
  gt_boxes[:,2:]=gt_boxes[:,:2]+gt_boxes[:,2:]
  gt_box=gt_boxes[id-1]
except IOError:
  print "Fail to open ", gt_path


frame_data=caffe.io.load_image(frame_path) #(432,576,3), in [0,1]
w,h,c=frame_data.shape
frame_shape=[c,w,h]
batch_size=1
net.blobs['data'].reshape(batch_size,c,w,h)
net.reshape()

transformer=caffe.io.Transformer({'data' :net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) #move channels to the outermost dimension
#mu=0.5
#transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255) #rescale from [0,1] to [0,255]
transformer.set_channel_swap('data', (2,1,0)) #swap channels from RGB to BGR
#(c,w,h)
transformed_frame=transformer.preprocess('data', frame_data)
plt.imshow(frame_data)
plt.show()
#set up net.blobs['data']
mu_channel=np.array([103.993,116.561,122.598],dtype=float)
mu_channel=mu_channel.reshape((3,1))
mu_channel=mu_channel[:,:,np.newaxis]
transformed_frame=transformed_frame-mu_channel

net.blobs['data'].data[...]=transformed_frame

#set up net.blobs['im_info']
print "Image Size: ",w,h
net.blobs['im_info'].data[...]=[c,w,h]

#set up net.blobs['rois']
#gt_box=np.array(gt_box,dtype=float)
box_shift=10
gt_box_frame=np.vstack([gt_box,gt_box-box_shift,gt_box+box_shift])
batch_ind=np.zeros((gt_box_frame.shape[0],1))
gt_box_frame=np.hstack([batch_ind, gt_box_frame])#row:(batch_id,x1,y1,x2,y2)
print "gt_box_frame is:"
print gt_box_frame
#restrict x1,y1,x2,y2
ind_zero=np.where(gt_box_frame<0)
ind=np.where(ind_zero[1]==1)
gt_box_frame[ind_zero[0][ind],1]=0 #x1=max(0,x1)
ind=np.where(ind_zero[1]==2)
gt_box_frame[ind_zero[0][ind],2]=0 #y1=max(0,y1)
ind=np.where(ind_zero[1]==3)
gt_box_frame[ind_zero[0][ind],3]=0 #x2=max(0,x1)
ind=np.where(ind_zero[1]==4)
gt_box_frame[ind_zero[0][ind],4]=0 #y2=max(0,y1)
ind_w=np.where(gt_box_frame>=w)
ind=np.where(ind_w[1]==1)
gt_box_frame[ind_w[0][ind],1]=w-1 #x1=min(w-1,x1)
ind=np.where(ind_w[1]==3)
gt_box_frame[ind_w[0][ind],3]=w-1 #x1=min(w-1,x2)
ind_h=np.where(gt_box_frame>=h)
ind=np.where(ind_h[1]==2)
gt_box_frame[ind_h[0][ind],2]=h-1 #x1=min(h-1,y1)
ind=np.where(ind_h[1]==4)
gt_box_frame[ind_h[0][ind],4]=h-1 #x1=min(h-1,y2)
print "gt_box_frame is:"
print gt_box_frame
net.blobs['rois'].data[...]=gt_box_frame

output=net.forward()
features=[]
for k,v in output.items():
  print k, v.shape
  features.append({k:v})
print type(features)
np.save("basketball_features.npy",features)
