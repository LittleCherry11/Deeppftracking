#particle filter implement
from filterpy.monte_carlo.resampling import *
import scipy.stats
import numpy as np
np.seed(7)

def create_particles(u,v,s,r,du,dv,ds,dr):
    '''(center_x,center_y,scale,aspect_ratio,...) first-order movement model'''
    N=8
    particles=np.empty((N,1))
#bbox: (x1,y1,x2,y2)
def bbox_to_states(bbox):
    '''(x1,y1,x2,y2)->(cx,cy,w,h)'''
def state_to_bbox(state):
    '''(cx,cy,w,h)->(x1,y1,x2,y2)'''
class PFfilter:
    def __init__(self,state):
        '''state:u,v,s,r,du,dv,ds,dr'''
        self.state0=state
        self.num_particles=50
        self.weights=np.ones((self.num_particles,1))/self.num_particles
        self.dt=0.1
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
        self.particles=np.random.multivariate_normal(self.state0,self.gaussian_cov,self.num_particles)#(N,8)
    def predict_particles(self):
        '''p(x_t|x_(t-1))'''
        self.particles[:,0:4]=self.particles[:,0:4]+self.dt*self.particles[:,4:]
        #add gaussian noise in state transition model
        self.particles+=np.random.multivariate_normal(np.zeros((1,8)),0.5*self.gaussian_cov,self.num_particles)
        #post precessing: restrict the states within image

    def update_particles(self,conf):
        '''set weights according to the conf: p(y_t|x_t)'''
        self.weights=conf
        self.weights+=1e-20#for numeric stable
        self.weights/=np.sum(self.weights)

    def estimate(self):
        '''estimate current position'''
        self.cur_pos=np.average(self.particles,weights=self.weights,axis=0)

    def neff(self):
        '''judge whether need resampling'''
        return 1./np.sum(np.square(self.weights))

    def resample(self,method='residual_resample'):
        '''
        resample using method, return indexes
        method: ressidual_resample,multinomial_resample,systematic_resmaple,stratified_resample
        '''
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

        return self.estimate()






