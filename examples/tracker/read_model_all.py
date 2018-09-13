import caffe
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
def main(args):
  root_path="/home/ccjiang/Documents/py-faster-rcnn/caffe-fast-rcnn/examples/tracker/"
  model_def=os.path.join(root_path, args.prototxt)
  model_weight=os.path.join(root_path,args.caffemodel)
  vis=args.vis
  debug=args.debug
  save=args.save

  caffe.set_mode_gpu()
  caffe.set_device(1)
  net=caffe.Net(model_def, model_weight, caffe.TEST)
  print "Initializing %s from %s"%(model_def,model_weight)
  if debug:
    print net.params.keys()
    print net.blobs.keys()


  dataset_path="/data/OTB50"
  sequence="Bird1"
  for t in os.walk(os.path.join(dataset_path,sequence,sequence,"img")):
    if t[0]==os.path.join(dataset_path,sequence,sequence,"img"):
      nFrame=len(t[2])
      print "Total frames are: ",nFrame



  gt_path=os.path.join(dataset_path,sequence,sequence,"groundtruth_rect.txt")

  try:
    gt_str=np.loadtxt(gt_path, dtype=str, delimiter='\n')
    gt_boxes=map(lambda x : map(float, x.split(',')), gt_str)
    gt_boxes=np.array(gt_boxes, dtype=float)
    gt_boxes[:,2:]=gt_boxes[:,:2]+gt_boxes[:,2:] #(x1,y1,w,h) -> (x1,y1,x2,y2)
    #gt_box=gt_boxes[id-1]
  except IOError:
    print "Fail to open ", gt_path


  for id in np.arange(0,nFrame):
    frame_name="img/%04d.jpg"%(id+1)
    print "Start processing: %s"%frame_name
    frame_path=os.path.join(dataset_path,sequence,sequence,frame_name)
    frame_data=caffe.io.load_image(frame_path) #(432,576,3), in [0,1]
    if id==0:
      w,h,c=frame_data.shape
      frame_shape=[c,w,h]
      batch_size=1
      net.blobs['data'].reshape(batch_size,c,w,h)
      net.reshape()
      # set up net.blobs['im_info']
      print "Image Size: ", w, h
      net.blobs['im_info'].data[...] = [c, w, h]
      transformer=caffe.io.Transformer({'data' :net.blobs['data'].data.shape})
      transformer.set_transpose('data', (2,0,1)) #move channels to the outermost dimension
      #mu=0.5
      #transformer.set_mean('data', mu)
      mu_channel = np.array([103.993, 116.561, 122.598], dtype=float)
      mu_channel = mu_channel.reshape((3, 1))
      mu_channel = mu_channel[:, :, np.newaxis]
      transformer.set_raw_scale('data', 255) #rescale from [0,1] to [0,255]
      transformer.set_channel_swap('data', (2,1,0)) #swap channels from RGB to BGR
    #(c,w,h)
    transformed_frame=transformer.preprocess('data', frame_data)
    if vis:
      plt.imshow(frame_data)
      plt.show()
    #set up net.blobs['data']

    transformed_frame=transformed_frame-mu_channel

    net.blobs['data'].data[...]=transformed_frame



    #set up net.blobs['rois']

    gt_box=gt_boxes[id]
    box_shift=10
    gt_box_frame=np.vstack([gt_box,gt_box+box_shift,gt_box-box_shift])
    batch_ind=np.zeros((gt_box_frame.shape[0],1))
    gt_box_frame=np.hstack([batch_ind, gt_box_frame])#row:(batch_id,x1,y1,x2,y2)
    print "gt_box_frame is:"
    print gt_box_frame
    #restrict x1,y1,x2,y2
    gt_box_frame[:, 1] = np.minimum(np.maximum(0, gt_box_frame[:, 1]), w)
    gt_box_frame[:, 3] = np.minimum(np.maximum(0, gt_box_frame[:, 3]), w)
    gt_box_frame[:, 2] = np.minimum(np.maximum(0, gt_box_frame[:, 2]), h)
    gt_box_frame[:, 4] = np.minimum(np.maximum(0, gt_box_frame[:, 4]), h)
    '''
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
    '''
    print "restricted gt_box_frame is:"
    print gt_box_frame
    net.blobs['rois'].data[...]=gt_box_frame

    output=net.forward()
    features=[]
    for k,v in output.items():
      print k, v.shape
      features.append({k:v})
    if debug:
      print type(features)
    if save:
      np.save("features/basketball_features_%d.npy"%(id+1),features)

if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument("--vis",action='store_true')
  parser.add_argument("--debug",action='store_true')
  parser.add_argument("--save",action='store_false')
  parser.add_argument("--prototxt",default="vgg16_align.prototxt",type=str)
  parser.add_argument("--caffemodel",default="VGG_ILSVRC_16_layers.caffemodel",type=str)
  args=parser.parse_args()
  main(args)
