import caffe
import numpy as np



class VGGnet:
    def __init__(self,model_def,model_weight):
        self.model_def=model_def
        self.model_weight=model_weight
        self.device=0#1
        print "Set_mode_gpu()..."
        caffe.set_mode_gpu()
        print "Set_device()..."
        caffe.set_device(self.device)

        self.net=caffe.Net(self.model_def,weights=self.model_weight,phase=caffe.TEST)
        print "Initializing %s from %s"%(self.model_def,self.model_weight)

        self.f3=[]
        self.f4=[]
        self.f5=[]
        self.label=[]
    def reshape(self,w,h,nbox,batch_size=1):
        c=3
        frame_shape=[c,w,h]
        self.net.blobs['data'].reshape(batch_size,c,h,w)
        nelem=5
        self.net.blobs['rois'].reshape(nbox,nelem)
        self.net.reshape()
        #print "Image Size: ",w,h
        self.net.blobs['im_info'].data[...]=[c,h,w]
        self.transformer=caffe.io.Transformer({'data':self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data',(2,0,1))#move channels to the outermost dimension
        mu_channel=np.array([103.993,116.561,122.598],dtype=float).reshape((3,1))#compute in ILSVRC12 224
        self.mu_channel=mu_channel[:,:,np.newaxis]#(3,1,1)
        self.transformer.set_mean('data',self.mu_channel)
        self.transformer.set_raw_scale('data',255)#rescale from [0,1] to [0,255]
        self.transformer.set_channel_swap('data',(2,1,0))#from RGB to BGR
    def get_features_first_raw(self,img,boxes_raw,id):
        '''img=caffe.io.load_image('path'), boxes:(nbox,4),already be restricted, id: frame id'''
        transformed_frame=self.transformer.preprocess('data',img)
        #transformed_frame=transformed_frame-self.mu_channel#have already add set_mean in transformer

        self.net.blobs['data'].data[...]=transformed_frame
        boxes=np.zeros((boxes_raw.shape[0],boxes_raw.shape[1]+1))
        boxes[:,1:]=boxes_raw
        self.net.blobs['rois'].data[...]=boxes
        nbox=boxes.shape[0]
        print "Start doing forward..."
        output=self.net.forward()
        print "Finish doing forward"
        f3=[]
        f4=[]
        f5=[]

        for k,v in output.items():
            if k=='feature3':
                f=np.split(v,v.shape[0],axis=0)
                f=map(lambda x:x.squeeze(),f)
                f3=f3+f#[f0,f1,f2,...]
            if k == 'feature4':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f4 = f4 + f
            if k == 'feature5':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f5 = f5 + f
        self.f3=np.array(f3).reshape((nbox,-1))#(nbox,256*7*7)
        self.f4=np.array(f4).reshape((nbox,-1))
        self.f5=np.array(f5).reshape((nbox,-1))
        res={'f3':self.f3,'f4':self.f4,'f5':self.f5}
        save=False
        if save:
            np.save('frame_%d.npy'%id,res)
        return res
    def get_features_second_raw(self,boxes_raw,id):
        '''img=caffe.io.load_image('path'), boxes:(nbox,4),already be restricted, id: frame id'''
        #transformed_frame = self.transformer.preprocess('data', img)
        # transformed_frame=transformed_frame-self.mu_channel#have already add set_mean in transformer

        #self.net.blobs['data'].data[...] = transformed_frame
        boxes = np.zeros((boxes_raw.shape[0], boxes_raw.shape[1] + 1))
        boxes[:, 1:] = boxes_raw
        self.net.blobs['rois'].data[...] = boxes
        nbox = boxes.shape[0]
        output = self.net.forward(start='roi_pool3')
        f3 = []
        f4 = []
        f5 = []

        for k, v in output.items():
            if k == 'feature3':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f3 = f3 + f  # [f0,f1,f2,...]
            if k == 'feature4':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f4 = f4 + f
            if k == 'feature5':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f5 = f5 + f
        self.f3 = np.array(f3).reshape((nbox, -1))  # (nbox,256*7*7)
        self.f4 = np.array(f4).reshape((nbox, -1))
        self.f5 = np.array(f5).reshape((nbox, -1))
        res = {'f3': self.f3, 'f4': self.f4, 'f5': self.f5}
        save = False
        if save:
            np.save('frame_%d.npy' % id, res)
        return res
    def get_features_first(self, img, boxes, id):
        '''img=caffe.io.load_image('path'), boxes:(nbox,5),already be restricted, id: frame id'''
        transformed_frame = self.transformer.preprocess('data', img)
        # transformed_frame=transformed_frame-self.mu_channel#have already add set_mean in transformer

        self.net.blobs['data'].data[...] = transformed_frame

        self.net.blobs['rois'].data[...] = boxes
        nbox = boxes.shape[0]
        output = self.net.forward()
        f3 = []
        f4 = []
        f5 = []

        for k, v in output.items():
            if k == 'feature3':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f3 = f3 + f  # [f0,f1,f2,...]
            if k == 'feature4':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f4 = f4 + f
            if k == 'feature5':
                f = np.split(v, v.shape[0], axis=0)
                f = map(lambda x: x.squeeze(), f)
                f5 = f5 + f
        self.f3 = np.array(f3).reshape((nbox, -1))  # (nbox,256*7*7)
        self.f4 = np.array(f4).reshape((nbox, -1))
        self.f5 = np.array(f5).reshape((nbox, -1))
        res = {'f3': self.f3, 'f4': self.f4, 'f5': self.f5}
        save = False
        if save:
            np.save('frame_%d.npy' % id, res)
        return res




