#particle filter implement
from filterpy.monte_carlo.resampling import *
import scipy.stats
import os
import argparse
import numpy as np
import cv2
np.random.seed(7)

eps=1e-10
#bbox: (x1,y1,x2,y2)
def bbox_to_states(bbox,area,ratio):
    '''(x1,y1,x2,y2)->(cx,cy,s,r,dcx,dcy)'''
    bbox=bbox.reshape((-1,4))
    #np.save('bbox.npy',bbox)
    box_w=bbox[:,2]-bbox[:,0]
    box_h=bbox[:,3]-bbox[:,1]
    cx=bbox[:,0]+box_w/2
    cy=bbox[:,1]+box_h/2
    s=box_w*box_h/area
    #print box_w/box_h
    r=np.log(box_w/box_h)/np.log(ratio)

    states=np.zeros((bbox.shape[0],6))
    states[:,0:4]=np.vstack((cx,cy,s,r)).transpose()

    return states
def state_to_bbox(state,area,ratio):
    '''(cx,cy,s,r,dcx,dcy)->(x1,y1,x2,y2)'''
    s=state[:,2]*area
    #print s
    r=np.power(ratio,state[:,3])
    #print r
    box_w=np.sqrt(r*s)
    #print box_w
    box_h=np.sqrt(s/(r+eps))
    #print box_h
    x1=state[:,0]-box_w/2
    y1=state[:,1]-box_h/2
    x2=state[:,0]+box_w/2
    y2=state[:,1]+box_h/2
    
    bbox=np.vstack((x1,y1,x2,y2)).transpose()
    #print bbox
    return bbox
def calc_iou(gt_box,pred_box):
    '''gt_box: (1,4)  pred_box: (N,4)'''
    gt_box=gt_box.reshape((1,4))
    iou=np.zeros((pred_box.shape[0],1))
    x1=np.maximum(gt_box[:,0],pred_box[:,0])
    x2=np.minimum(gt_box[:,2],pred_box[:,2])
    y1=np.maximum(gt_box[:,1],pred_box[:,1])
    y2=np.minimum(gt_box[:,3],pred_box[:,3])
    intersection=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    union=(gt_box[:,2]-gt_box[:,0])*(gt_box[:,3]-gt_box[:,1])+(pred_box[:,2]-pred_box[:,0])*(pred_box[:,3]-pred_box[:,1])
    intersection=intersection.astype(np.float)
    iou=intersection/(union-intersection)
    return iou
class PFfilter:
    def __init__(self,state,area,ratio):
        '''state:u,v,s,r,du,dv'''
        self.state0=state
        self.area=area
        self.ratio=ratio
        self.prev_pos=state_to_bbox(state,area,ratio)
        print 'prev_pos: ',self.prev_pos.shape,' ',self.prev_pos.dtype
        self.cur_pos=np.empty((1,4),dtype=np.float)
        print 'cur_pos ',self.cur_pos.shape,' ',self.cur_pos.dtype
        np.copyto(self.cur_pos,self.prev_pos)
        #self.cur_pos=self.prev_pos
        #print 'cur_pos.shape= ',self.cur_pos.shape
        print 'initial state is: ',self.state0
        self.num_particles=200
        self.weights=np.ones((self.num_particles,))/self.num_particles
        self.dt=1
        self.gaussian_cov=np.array([[10,0,0,0,0,0,0,0],
                                    [0,10,0,0,0,0,0,0],
                                    [0,0,0.01,0,0,0,0,0],
                                    [0,0,0,0.01,0,0,0,0],
                                    [0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,1,0,0],
                                    [0,0,0,0,0,0,0.001,0],
                                    [0,0,0,0,0,0,0,0.001]])
    def create_particles(self):
        '''cast particles around state0'''
        Q=[[5,0],[0,5]]
        QL=np.linalg.cholesky(Q)
        self.particles=np.empty((self.num_particles,6))
        #cx=cx+dt*dcx+noise
        #print self.state0.shape
        #print self.particles.shape
        c=self.state0[:,:2]+self.state0[:,4:]
        #print c.shape
        d=np.dot(QL,np.random.randn(2,self.num_particles)).transpose()
        #print d.shape
        self.particles[:,:2]=self.state0[:,:2]+self.dt*self.state0[:,4:]+np.dot(QL,np.random.randn(2,self.num_particles)).transpose()
        #s
        a=1.5
        dsn=np.random.randn(self.num_particles)
        ds=np.power(a,dsn)
        self.particles[:,2]=self.state0[:,2]*ds
        #r
        R=0.16
        dr=np.random.randn(self.num_particles)*np.sqrt(R)+1
        self.particles[:,3]=self.state0[:,3]*dr
        #dcx,dcy
        self.particles[:,4:]=np.dot(QL,np.random.randn(2,self.num_particles)).transpose()


        #self.particles=np.random.multivariate_normal(self.state0.squeeze(),self.gaussian_cov,self.num_particles)#(N,8)
    def predict_particles(self):
        '''p(x_t|x_(t-1))'''
        #self.particles[:,0:4]=self.particles[:,0:4]+self.dt*self.particles[:,4:]
        #add gaussian noise in state transition model
        #self.particles+=np.random.multivariate_normal(np.zeros((8,)),0.5*self.gaussian_cov,self.num_particles)
        Q=[[10,0],[0,10]]
        QL=np.linalg.cholesky(Q)
        self.particles[:,:2]=self.particles[:,:2]+self.dt*self.particles[:,4:]+np.dot(QL,np.random.randn(2,self.num_particles)).transpose()
        a=1.5
        dsn=np.random.randn(self.num_particles)
        ds=np.power(a,dsn)
        self.particles[:,2]=self.particles[:,2]*ds

        R=0.16
        dr=np.random.randn(self.num_particles)*np.sqrt(R)+1
        self.particles[:,3]=self.particles[:,3]*dr
        #dcx,dcy
        V=[[5,0],[0,5]]
        VL=np.linalg.cholesky(V)
        dxy=self.cur_pos.reshape((1,4))[:,:2]-self.prev_pos.reshape((1,4))[:,:2]
        print 'dxy = ',dxy
        self.particles[:, 4:] =dxy+ np.dot(VL, np.random.randn(2,self.num_particles)).transpose()


        #self.restrict_particles()
        #post precessing: restrict the states within image
    def restrict_particles(self,w,h):
        bboxes=state_to_bbox(self.particles,self.area,self.ratio)
        # restrict x1,y1,x2,y2
        bboxes[:,0]=np.minimum(np.maximum(0,bboxes[:,0]),w)
        bboxes[:,2] = np.minimum(np.maximum(0, bboxes[:, 2]), w)
        bboxes[:,1] = np.minimum(np.maximum(0, bboxes[:, 1]), h)
        bboxes[:,3] = np.minimum(np.maximum(0, bboxes[:, 3]), h)
        #prev_particles= self.particles
        state_half=bbox_to_states(bboxes,self.area,self.ratio)
        self.particles[:,:4]=state_half[:,:4]
    def update_particles(self,conf):
        '''set weights according to the conf: p(y_t|x_t)'''
        self.weights=conf
        self.weights+=eps#for numeric stable
        self.weights/=np.sum(self.weights)

    def estimate(self):
        '''estimate current position'''
        np.copyto(self.prev_pos,self.cur_pos)
        #self.prev_pos=self.cur_pos
        cur_pos=np.average(self.particles,weights=self.weights.squeeze(),axis=0)#(cx,cy,s,r,dcx,dcy)
        #cur_pos=state_to_bbox(cur_pos,self.area,self.ratio)
        self.cur_pos[0,0]=np.minimum(np.maximum(0,cur_pos[0]),w)
        self.cur_pos[0,1]=np.minimum(np.maximum(0,cur_pos[1]),h)
        #self.cur_pos[0,2]=np.minimum(np.maximum(0,cur_pos[2]),w)
        #self.cur_pos[0,3]=np.minimum(np.maximum(0,cur_pos[3]),h)

    def neff(self):
        '''judge whether need resampling'''
        eff=1./np.sum(np.square(self.weights))
        print 'eff is ',eff
        return eff

    def resample(self,method='residual_resample'):
        '''
        resample using method, return indexes
        method: ressidual_resample,multinomial_resample,systematic_resmaple,stratified_resample
        '''
        print "resampling..."
        indexes=residual_resample(self.weights)
        self.particles[:]=self.particles[indexes]
        self.weights[:]=self.weights[indexes]
        self.weights/=np.sum(self.weights)

    def run(self,conf):
        self.predict_particles()
        #other process to get conf
        self.update_particles(conf)
        if self.neff()<len(self.particles)/2:
            self.resample()
        self.estimate()
        return self.cur_pos




if __name__=='__main__':
    root_path = "/home/ccjiang/Documents/py-faster-rcnn/caffe-fast-rcnn/examples/tracker/"

    #vis = args.vis
    #debug = args.debug
    #save = args.save


    dataset_path = "/data/OTB50"
    sequence = "Basketball"
    for t in os.walk(os.path.join(dataset_path, sequence, "img")):
        if t[0] == os.path.join(dataset_path, sequence, "img"):
            nFrame = len(t[2])
            print "Total frames are: ", nFrame

    gt_path = os.path.join(dataset_path, sequence, "groundtruth_rect.txt")

    try:
        gt_str = np.loadtxt(gt_path, dtype=str, delimiter='\n')
        gt_boxes = map(lambda x: map(float, x.split(',')), gt_str)
        gt_boxes = np.array(gt_boxes, dtype=float)
        gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:]  # (x1,y1,w,h) -> (x1,y1,x2,y2)
        # gt_box=gt_boxes[id-1]
    except IOError:
        print "Fail to open ", gt_path

    for id in np.arange(0, 100):
        frame_name = "img/%04d.jpg" % (id + 1)
        print "Start processing: %s" % frame_name
        frame_path = os.path.join(dataset_path, sequence, frame_name)
        frame_data = cv2.imread(frame_path)  # (432,576,3), in [0,1]
        gt_box=gt_boxes[id]
        print 'ground truth bbox is: ',gt_box
        if id == 0:
            w, h, c = frame_data.shape
            frame_shape = [c, w, h]
            area=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
            ratio=(gt_box[2]-gt_box[0])/(gt_box[3]-gt_box[1]) #ratio=w/h
            # set up net.blobs['im_info']
            print "Image Size: ", w, h

            filter=PFfilter(bbox_to_states(gt_box,area,ratio),area,ratio)
            filter.create_particles()

        cv2.rectangle(frame_data,(int(gt_box[0]),int(gt_box[1])),(int(gt_box[2]),int(gt_box[3])),(255,0,0),2,1)
        show_particles=0
        if show_particles:
            for i in range(filter.num_particles):
                cx=filter.particles[i,0]
                cy=filter.particles[i,1]
                cv2.circle(frame_data,(int(cx),int(cy)),1,(0,255,0),thickness=1)
        filter.predict_particles()
        #np.save('particles.npy',filter.particles)
        filter.restrict_particles(w,h)
        if show_particles:
            for i in range(filter.num_particles):
                cx=filter.particles[i,0]
                cy=filter.particles[i,1]
                cv2.circle(frame_data,(int(cx),int(cy)),1,(0,0,255),thickness=1)
        #compute conf
        conf=np.zeros(filter.weights.shape)
        #np.save('particles.npy',filter.particles)
        pred_boxes=state_to_bbox(filter.particles,area,ratio)

        #print 'pred_boxes: ',pred_boxes
        #for i in range(conf.shape[0]):
        #    pred_box=pred_boxes[i,:]
            #print "pred_box is: ",pred_box
            #conf[i]=np.dot(gt_box,pred_box)/np.linalg.norm(gt_box,ord=2)
        #    conf[i]=np.dot(gt_box,pred_box)/np.sum(np.square(gt_box))
        conf=calc_iou(gt_box,pred_boxes)
        print 'conf is: ',conf
        filter.update_particles(conf)
        if filter.neff()<len(filter.particles)*2/3: #1/2
            filter.resample()
        filter.estimate()

        #(B,G,R)
        cv2.circle(frame_data, (int(filter.cur_pos[0,0]), int(filter.cur_pos[0,1])), 2, (0, 0, 255), thickness=1)
        cv2.imshow(sequence,frame_data)
        c=cv2.waitKey()
        print 'You press: ',chr(c)
        if chr(c)=='c':
            break

        '''
        print "gt_box_is:"
        print gt_box
        # restrict x1,y1,x2,y2
        gt_box[:,0]=np.minimum(np.maximum(0,gt_box[:,0]),w)
        gt_box[:,2] = np.minimum(np.maximum(0, gt_box[:, 2]), w)
        gt_box[:,1] = np.minimum(np.maximum(0, gt_box[:, 1]), h)
        gt_box[:,3] = np.minimum(np.maximum(0, gt_box[:, 3]), h)

        #ind_zero = np.where(gt_box_frame < 0)
        #ind = np.where(ind_zero[1] == 1)
        #gt_box_frame[ind_zero[0][ind], 1] = 0  # x1=max(0,x1)
        #ind = np.where(ind_zero[1] == 2)
        #gt_box_frame[ind_zero[0][ind], 2] = 0  # y1=max(0,y1)
        #ind = np.where(ind_zero[1] == 3)
        #gt_box_frame[ind_zero[0][ind], 3] = 0  # x2=max(0,x1)
        #ind = np.where(ind_zero[1] == 4)
        #gt_box_frame[ind_zero[0][ind], 4] = 0  # y2=max(0,y1)
        #ind_w = np.where(gt_box_frame >= w)
        #ind = np.where(ind_w[1] == 1)
        #gt_box_frame[ind_w[0][ind], 1] = w - 1  # x1=min(w-1,x1)
        #ind = np.where(ind_w[1] == 3)
        #gt_box_frame[ind_w[0][ind], 3] = w - 1  # x1=min(w-1,x2)
        #ind_h = np.where(gt_box_frame >= h)
        #ind = np.where(ind_h[1] == 2)
        #gt_box_frame[ind_h[0][ind], 2] = h - 1  # x1=min(h-1,y1)
        #ind = np.where(ind_h[1] == 4)
        #gt_box_frame[ind_h[0][ind], 4] = h - 1  # x1=min(h-1,y2)
        print "restricted gt_box_frame is:"
        print gt_box
        '''



