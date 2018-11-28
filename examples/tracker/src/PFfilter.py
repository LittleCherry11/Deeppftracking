# particle filter implement
from filterpy.monte_carlo.resampling import *
import scipy.stats
import os
import argparse
import numpy as np
import cv2
import re
import utils
import time
import random

np.random.seed(int(time.time()))  # 7

eps = 1e-10


class PFfilter:
    def __init__(self, state, area, ratio, img_w, img_h, N=256):
        '''state:u,v,s,r,du,dv'''
        self.state0 = state
        self.area = area
        self.ratio = ratio
        self.width = img_w  # img width,never change
        self.height = img_h  # img_height, never change
        # prev box position: [[x1,y1,x2,y2]]
        self.prev_pos = utils.state_to_bbox(state, area, ratio)
        self.prev_c = np.array(
            [(self.prev_pos[0, 2] + self.prev_pos[0, 0]) / 2, (self.prev_pos[0, 3] + self.prev_pos[0, 1]) / 2],
            dtype=np.float32)
        # print 'prev_pos: ',self.prev_pos.shape,' ',self.prev_pos.dtype
        # current box position: [[x1,x2,y1,y2]]
        self.cur_pos = np.empty((1, 4), dtype=np.float32)

        # print 'cur_pos ',self.cur_pos.shape,' ',self.cur_pos.dtype
        np.copyto(self.cur_pos, self.prev_pos)
        # current center,[cx,cy]
        self.cur_c = np.array(
            [(self.cur_pos[0, 2] + self.cur_pos[0, 0]) / 2, (self.cur_pos[0, 3] + self.cur_pos[0, 1]) / 2],
            dtype=np.float32)

        self.prev_a = 1.0
        self.cur_a = 1.0
        self.box_w = self.cur_pos[0, 2] - self.cur_pos[0, 0]  # current box width, update by estimate()
        self.box_h = self.cur_pos[0, 3] - self.cur_pos[0, 1]  # current box height, update by estimate()
        self.cur_a = self.box_h * self.box_w / self.area

        # print 'initial state is: ', self.state0
        self.num_particles = N
        self.weights = np.ones((self.num_particles,)) / self.num_particles
        self.dt = 1.0

    def reset(self, state, area, ratio):
        '''state:u,v,s,r,du,dv'''
        self.state0 = state
        self.area = area
        self.ratio = ratio

        # cur box position: [[x1,y1,x2,y2]]
        self.cur_pos = utils.state_to_bbox(state, area, ratio)
        self.cur_c = np.array(
            [(self.prev_pos[0, 2] + self.prev_pos[0, 0]) / 2, (self.prev_pos[0, 3] + self.prev_pos[0, 1]) / 2],
            dtype=np.float32)

        self.box_w = self.cur_pos[0, 2] - self.cur_pos[0, 0]  # current box width, update by estimate()
        self.box_h = self.cur_pos[0, 3] - self.cur_pos[0, 1]  # current box height, update by estimate()
        self.cur_a = self.box_h * self.box_w / self.area
        # self.cur_pos=self.prev_pos
        # print 'cur_pos.shape= ',self.cur_pos.shape

        self.weights = np.ones((self.num_particles,)) / self.num_particles
        D = np.array([[self.box_w, 0], [0, self.box_h]], dtype=np.float32)
        vs = 0.01  # 2
        VL = D * vs
        dxy = self.cur_c.reshape((1, 2))[:, :2] - self.prev_c.reshape((1, 2))[:, :2]

        maxd = self.area * 0.01
        # dxy cannot be too much
        area_d = np.abs(dxy[:, 0:1]) * np.abs(dxy[:, 1:])
        ind = np.where(area_d > maxd)
        dxy[ind[0], :] = dxy[ind[0], :] / np.sqrt(area_d[ind[0], :]) * np.sqrt(maxd)
        # print 'dxy = ', dxy
        self.particles[:, 4:] = dxy + np.dot(VL, np.random.randn(2, self.num_particles)).transpose()

        Q = 0.02  # 0.02
        QL = D * Q
        # cx,cy:cx=cx+dt*dcx+noise
        self.particles[:, :2] = self.state0[:, :2] + self.dt * self.state0[:, 4:] + np.dot(QL, np.random.randn(2,
                                                                                                               self.num_particles)).transpose()
        self.mltply = False
        if self.mltply:
            # s
            a = 1.5
            dsn = np.random.randn(self.num_particles)
            ds = np.power(a, dsn)
            self.particles[:, 2] = self.state0[:, 2] * ds
            # r
            R = 0.16
            dr = np.random.randn(self.num_particles) * np.sqrt(R) + 1
            self.particles[:, 3] = self.state0[:, 3] * dr
        else:  # add predict
            ca = 0.0001  # 0.01
            cr = 0.00001
            self.particles[:, 2] = 1 + np.random.randn(self.num_particles) * np.sqrt(ca)
            self.particles[:, 3] = 1 + np.random.randn(self.num_particles) * np.sqrt(cr)

    def create_particles(self):
        '''cast particles around state0'''
        self.particles = np.empty((self.num_particles, 6))
        # D = np.array([[self.ratio * self.ratio, 0], [0, 1]], dtype=np.float32)
        # dcx,dcy
        # V = [[2, 0], [0, 2]]
        # vs = 2
        # V = D * vs
        # VL = np.linalg.cholesky(V)
        D = np.array([[self.box_w, 0], [0, self.box_h]], dtype=np.float32)
        vs = 0.01  # 2

        VL = D * vs
        dxy = self.cur_c.reshape((1, 2))[:, :2] - self.prev_c.reshape((1, 2))[:, :2]

        maxd = self.area * 0.01
        # dxy cannot be too much
        area_d = np.abs(dxy[:, 0:1]) * np.abs(dxy[:, 1:])
        ind = np.where(area_d > maxd)
        dxy[ind[0], :] = dxy[ind[0], :] / np.sqrt(area_d[ind[0], :]) * np.sqrt(maxd)
        # print 'dxy = ', dxy
        self.particles[:, 4:] = dxy + np.dot(VL, np.random.randn(2, self.num_particles)).transpose()

        # Q = [[5, 0], [0, 5]]
        # qs=4
        # Q=D*qs
        # QL = np.linalg.cholesky(Q)
        # D = np.array([[self.box_w, 0], [0, self.box_h]])
        Q = 0.02
        QL = D * Q
        # cx,cy:cx=cx+dt*dcx+noise
        self.particles[:, :2] = self.state0[:, :2] + self.dt * self.state0[:, 4:] + np.dot(QL, np.random.randn(2,
                                                                                                               self.num_particles)).transpose()
        self.mltply = False
        if self.mltply:
            # s
            a = 1.5
            dsn = np.random.randn(self.num_particles)
            ds = np.power(a, dsn)
            self.particles[:, 2] = self.state0[:, 2] * ds
            # r
            R = 0.16
            dr = np.random.randn(self.num_particles) * np.sqrt(R) + 1
            self.particles[:, 3] = self.state0[:, 3] * dr
        else:  # add predict
            ca = 0.0001  # 0.01
            cr = 0.00001
            self.particles[:, 2] = 1 + np.random.randn(self.num_particles) * np.sqrt(ca)
            self.particles[:, 3] = 1 + np.random.randn(self.num_particles) * np.sqrt(cr)

        # self.particles[:,4:]=np.dot(QL,np.random.randn(2,self.num_particles)).transpose()

        # self.particles=np.random.multivariate_normal(self.state0.squeeze(),self.gaussian_cov,self.num_particles)#(N,8)

    def predict_particles(self, Q=0.02, cr=0.01, ca=0.001):
        '''p(x_t|x_(t-1)),ca:area, cr:ratio'''
        # update dcx,dcy
        # D = np.array([[self.ratio * self.ratio, 0], [0, 1]], dtype=np.float32)
        D = np.array([[self.box_w, 0], [0, self.box_h]], dtype=np.float32)
        vs = 0.01  # 2
        VL = D * vs
        # VL = np.linalg.cholesky(V)
        dxy = self.cur_c.reshape((1, 2))[:, :2] - self.prev_c.reshape((1, 2))[:, :2]
        maxd = self.area * 0.05  # 0.05
        # dxy cannot be too much
        area_d = np.abs(dxy[:, 0:1]) * np.abs(dxy[:, 1:])
        ind = np.where(area_d > maxd)
        dxy[ind[0], :] = dxy[ind[0], :] / np.sqrt(area_d[ind[0], :]) * np.sqrt(maxd)
        # print 'dxy = ', dxy
        self.particles[:, 4:] = dxy + np.dot(VL, np.random.randn(2, self.num_particles)).transpose()
        # self.particles[:,0:4]=self.particles[:,0:4]+self.dt*self.particles[:,4:]
        # add gaussian noise in state transition model
        # self.particles+=np.random.multivariate_normal(np.zeros((8,)),0.5*self.gaussian_cov,self.num_particles)
        # Q = [[10, 0], [0, 10]]
        # qs=10
        # Q=D*qs
        # QL = np.linalg.cholesky(Q)

        # Q = 0.04#0.02
        QL = D * Q
        self.particles[:, :2] = self.particles[:, :2] + self.dt * self.particles[:, 4:] + np.dot(QL, np.random.randn(2,
                                                                                                                     self.num_particles)).transpose()
        if self.mltply:
            a = 1.5
            dsn = np.random.randn(self.num_particles)
            ds = np.power(a, dsn)
            self.particles[:, 2] = self.particles[:, 2] * ds

            R = 0.16
            dr = np.random.randn(self.num_particles) * np.sqrt(R) + 1
            self.particles[:, 3] = self.particles[:, 3] * dr
        else:
            # cr = 0.01#0.001
            # ca=0.001#0.0001
            # print 'self.prev_a: ',self.prev_a
            # print 'self.cur_a: ',self.cur_a
            area_change = self.cur_a - self.prev_a
            # print 'area change: ',area_change
            area_change = 0
            self.particles[:, 2] = 1 + area_change + np.random.randn(self.num_particles) * np.sqrt(ca)
            self.particles[:, 2] = np.maximum(eps, self.particles[:, 2])
            self.particles[:, 3] = 1 + np.random.randn(self.num_particles) * np.sqrt(cr)
            self.particles[:, 3] = np.maximum(eps, self.particles[:, 3])

        # self.restrict_particles()
        # post precessing: restrict the states within image

    def restrict_particles(self, w, h):
        if self.mltply:
            bboxes = utils.state_to_bbox_m(self.particles, self.area, self.ratio)
        else:
            bboxes = utils.state_to_bbox(self.particles, self.area, self.ratio)
        # restrict x1,y1,x2,y2
        # bboxes[:, 0] = np.minimum(np.maximum(0, bboxes[:, 0]), w)
        # bboxes[:, 2] = np.minimum(np.maximum(0, bboxes[:, 2]), w)
        # bboxes[:, 1] = np.minimum(np.maximum(0, bboxes[:, 1]), h)
        # bboxes[:, 3] = np.minimum(np.maximum(0, bboxes[:, 3]), h)
        if self.mltply:
            bboxes = utils.restrict_box_m(bboxes, w, h)
        else:
            bboxes = utils.restrict_box(bboxes, w, h)
        # prev_particles= self.particles
        if self.mltply:
            state_half = utils.bbox_to_states_m(bboxes, self.area, self.ratio)
        else:
            state_half = utils.bbox_to_states(bboxes, self.area, self.ratio)
        self.particles[:, :4] = state_half[:, :4]
    def restrict_particles_extern(self,states,w,h):
        if self.mltply:
            bboxes = utils.state_to_bbox_m(states, self.area, self.ratio)
        else:
            bboxes = utils.state_to_bbox(states, self.area, self.ratio)
        # restrict x1,y1,x2,y2
        # bboxes[:, 0] = np.minimum(np.maximum(0, bboxes[:, 0]), w)
        # bboxes[:, 2] = np.minimum(np.maximum(0, bboxes[:, 2]), w)
        # bboxes[:, 1] = np.minimum(np.maximum(0, bboxes[:, 1]), h)
        # bboxes[:, 3] = np.minimum(np.maximum(0, bboxes[:, 3]), h)
        if self.mltply:
            bboxes = utils.restrict_box_m(bboxes, w, h)
        else:
            bboxes = utils.restrict_box(bboxes, w, h)
        # prev_particles= self.particles
        if self.mltply:
            state_half = utils.bbox_to_states_m(bboxes, self.area, self.ratio)
        else:
            state_half = utils.bbox_to_states(bboxes, self.area, self.ratio)
        return state_half
    def update_particles(self, conf):
        '''set weights according to the conf: p(y_t|x_t)'''
        # print 'conf is: ',conf
        self.weights *= conf
        self.weights += eps  # for numeric stable
        self.weights /= np.sum(self.weights)

    def estimate(self, k=10):
        '''estimate current position'''
        np.copyto(self.prev_pos, self.cur_pos)
        np.copyto(self.prev_c, self.cur_c)
        cur_pos = np.zeros((6,), dtype=np.float32)
        # there are two methods to estimating cur_pos: average or max
        # avreage
        maxw = np.max(self.weights)
        inds = np.where(self.weights>0.5*maxw)[0]
        cur_pos = np.average(self.particles[inds], weights=self.weights.squeeze()[inds], axis=0)  # (cx,cy,s,r,dcx,dcy)

        # max
        # cur_pos = self.particles[np.argmax(self.weights)]

        # max k pos
        '''
        #k = 3

        #print type(self.weights), self.weights.shape
        #print 'max: ',np.max(self.weights)
        #print 'min: ',np.min(self.weights)
        sort_ind = np.argsort(-self.weights.squeeze())
        cur_pos = np.average(self.particles[sort_ind[:k]], weights=self.weights[sort_ind[:k]], axis=0)
        '''
        '''
        #hist estimate
        count_xy,edge_x,edge_y=np.histogram2d(self.particles[:,0],self.particles[:,1],bins=40,weights=self.weights.squeeze())
        top3=(-count_xy).argsort(axis=None)[:2]
        ind_x=top3[:]/count_xy.shape[1]
        ind_y=top3[:]%count_xy.shape[1]
        if abs(max(ind_x)-min(ind_x))==1:
            #adjacent
            ind_right=max(ind_x)
            ind_left=min(ind_x)
            edge_x1=edge_x[ind_left]
            edge_x2=edge_x[ind_right+1]
        else:
            edge_x1=edge_x[ind_x[0]]
            edge_x2=edge_x[ind_x[0]+1]
        if abs(max(ind_y)-min(ind_y))==1:
            #adjacent
            ind_right = max(ind_y)
            ind_left = min(ind_y)
            edge_y1=edge_y[ind_left]
            edge_y2=edge_y[ind_right+1]
        else:
            edge_y1=edge_y[ind_y[0]]
            edge_y2=edge_y[ind_y[0]+1]

        cur_pos[0]=(edge_x1+edge_x2)/2.0
        cur_pos[1]=(edge_y1+edge_y2)/2.0
        #area and ratio
        count_sr, edge_s, edge_r = np.histogram2d(self.particles[:, 2], self.particles[:, 3], bins=20,
                                                  weights=self.weights.squeeze())
        top3 = (-count_sr).argsort(axis=None)[:2]
        ind_s = top3[:] / count_sr.shape[1]
        ind_r = top3[:] % count_sr.shape[1]
        if abs(max(ind_s) - min(ind_s)) == 1:
            # adjacent
            ind_right = max(ind_s)
            ind_left = min(ind_s)
            edge_s1 = edge_s[ind_left]
            edge_s2 = edge_s[ind_right + 1]
        else:
            edge_s1 = edge_s[ind_s[0]]
            edge_s2 = edge_s[ind_s[0] + 1]
        if abs(max(ind_r) - min(ind_r)) == 1:
            # adjacent
            ind_right = max(ind_r)
            ind_left = min(ind_r)
            edge_r1 = edge_r[ind_left]
            edge_r2 = edge_r[ind_right + 1]
        else:
            edge_r1 = edge_r[ind_r[0]]
            edge_r2 = edge_r[ind_r[0] + 1]

        cur_pos[2] = (edge_s1 + edge_s2) / 2.0
        cur_pos[3] = (edge_r1 + edge_r2) / 2.0
        '''
        if self.mltply:
            self.cur_pos = utils.state_to_bbox_m(cur_pos, self.area, self.ratio)
        else:
            self.cur_pos = utils.state_to_bbox(cur_pos, self.area, self.ratio)
        self.cur_pos = utils.restrict_box(self.cur_pos, self.width, self.height)
        self.box_h = self.cur_pos[0, 3] - self.cur_pos[0, 1]
        self.box_w = self.cur_pos[0, 2] - self.cur_pos[0, 0]
        self.cur_c[0] = np.minimum(np.maximum(0, cur_pos[0]), self.width)
        self.cur_c[1] = np.minimum(np.maximum(0, cur_pos[1]), self.height)
        # update self.cur_a and self.prev_a
        self.prev_a = self.cur_a
        self.cur_a = self.box_w * self.box_h / self.area
        # self.cur_pos[0, 2] = np.minimum(np.maximum(0, cur_pos[2]), self.width)
        # self.cur_pos[0, 3] = np.minimum(np.maximum(0, cur_pos[3]), self.height)
        # print 'prev_pos = ', self.prev_pos
        # print 'cur_pos = ', self.cur_pos

        # calculate s and r
        s = self.particles[:, 2]
        r = self.particles[:, 3]

        return cur_pos, s, r

    def neff(self):
        '''judge whether need resampling'''
        eff = 1. / np.sum(np.square(self.weights))
        # print 'eff is ', eff
        return eff
    def estimate_const(self,conf,k=10):
        cur_pos = np.zeros((6,), dtype=np.float32)

        # avreage
        cur_pos = np.average(self.particles, weights=conf, axis=0)  # (cx,cy,s,r,dcx,dcy)

        if self.mltply:
            cur_pos = utils.state_to_bbox_m(cur_pos, self.area, self.ratio)
        else:
            cur_pos = utils.state_to_bbox(cur_pos, self.area, self.ratio)
        cur_pos = utils.restrict_box(cur_pos, self.width, self.height)
        return cur_pos

    def resample(self, method='residual_resample'):
        '''
        resample using method, return indexes
        method: residual_resample,multinomial_resample,systematic_resmaple,stratified_resample
        '''
        # print "resampling..."
        indexes = residual_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        #self.weights[:] = self.weights[indexes]
        self.weights[:] = 1.0
        self.weights /= np.sum(self.weights)

    def sample_iou(self, gt_box, Q, T, R, N, thre_min=0, thre_max=1):

        sample_boxN = []
        sample_iouN = []
        cur_n = 0
        D = np.array([[self.box_w, 0], [0, self.box_h]])
        QL = D * Q
        # QL = np.linalg.cholesky(Q)
        a = 1.5  # 1.5

        sample_times = 0
        chg_i = 0
        while cur_n < N:
            sample_particles = np.zeros((self.num_particles, 6), dtype=np.float32)
            QL = D * Q
            sample_particles[:, :2] = self.particles[:, :2] + np.dot(QL,
                                                                     np.random.randn(2, self.num_particles)).transpose()

            if self.mltply:
                dsn = np.random.randn(self.num_particles) * T
                ds = np.power(a, dsn)
            else:
                ds = 1 + np.random.randn(self.num_particles) * T
                ds = np.maximum(0.01, ds)  # in case of ds<0
            sample_particles[:, 2] = self.particles[:, 2] * ds

            if self.mltply:
                dr = np.random.randn(self.num_particles) * R + 1
            else:
                dr = 1 + np.random.randn(self.num_particles) * R
                dr = np.maximum(0.01, dr)  # in case of dr<0
            sample_particles[:, 3] = self.particles[:, 3] * dr

            # get box
            if self.mltply:
                sample_box = utils.state_to_bbox_m(sample_particles, self.area, self.ratio)
            else:
                sample_box = utils.state_to_bbox(sample_particles, self.area, self.ratio)
            sample_box = utils.restrict_box(sample_box, self.width, self.height)
            # compute iou
            sample_iou = utils.calc_iou(gt_box, sample_box)
            # restrict iou
            ind = np.where((sample_iou >= thre_min) & (sample_iou <= thre_max))
            sample_box = sample_box[ind[0]]
            sample_iou = sample_iou[ind[0]]
            cur_n += sample_box.shape[0]
            sample_boxN.append(sample_box)
            sample_iouN.append(sample_iou.reshape((-1, 1)))
            if ind[0].shape[0] < self.num_particles / 2 and thre_max >= 0.8:
                if chg_i == 0:
                    Q *= 0.5
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 0.5
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 0.5
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            if ind[0].shape[0] < self.num_particles / 2 and thre_min <= 0.5:
                if chg_i == 0:
                    Q *= 2
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 2
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 2
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            sample_times += 1
            if sample_times >= 200:  # and cur_n>N/2.0:#100
                # print "Caution: too many loops in sampling"
                #break
                raise OverflowError()
        if cur_n >= N:
            sample_boxN = np.vstack(sample_boxN)[:N, :]
            sample_iouN = np.vstack(sample_iouN)[:N, :]
        else:


            sample_iouN = np.vstack(sample_iouN)
            sample_boxN = np.vstack(sample_boxN)

            diff_n = N - cur_n
            diff_ind = random.sample(range(cur_n), diff_n)  # need to ensure diff_n<cur_n
            sample_boxN = np.vstack([sample_boxN, sample_boxN[diff_ind]])
            sample_iouN = np.vstack([sample_iouN, sample_iouN[diff_ind]])
        return sample_boxN, sample_iouN
    def sample_iou_pred_box(self,pred_box,Q,T,R,N,thre_min=0,thre_max=1):
        #pred_box:np.array with shape(1,4)
        state_tmp=utils.bbox_to_states(pred_box,self.area,self.ratio)

        sample_boxN = []
        sample_iouN = []
        cur_n = 0
        D = np.array([[self.box_w, 0], [0, self.box_h]])
        QL = D * Q
        # QL = np.linalg.cholesky(Q)
        a = 1.5  # 1.5

        sample_times = 0
        chg_i = 0
        while cur_n < N:
            sample_particles = state_tmp.repeat(N, axis=0)
            QL = D * Q
            sample_particles[:, :2] += np.dot(QL,np.random.randn(2, N)).transpose()

            if self.mltply:
                dsn = np.random.randn(N) * T
                ds = np.power(a, dsn)
            else:
                ds = 1 + np.random.randn(N) * T
                ds = np.maximum(0.01, ds)  # in case of ds<0
            sample_particles[:, 2] *= ds

            if self.mltply:
                dr = np.random.randn(N) * R + 1
            else:
                dr = 1 + np.random.randn(N) * R
                dr = np.maximum(0.01, dr)  # in case of dr<0
            sample_particles[:, 3] *= dr

            # get box
            if self.mltply:
                sample_box = utils.state_to_bbox_m(sample_particles, self.area, self.ratio)
            else:
                sample_box = utils.state_to_bbox(sample_particles, self.area, self.ratio)
            sample_box = utils.restrict_box(sample_box, self.width, self.height)
            # compute iou
            sample_iou = utils.calc_iou(pred_box, sample_box)
            # restrict iou
            ind = np.where((sample_iou >= thre_min) & (sample_iou <= thre_max))
            sample_box = sample_box[ind[0]]
            sample_iou = sample_iou[ind[0]]
            cur_n += sample_box.shape[0]
            sample_boxN.append(sample_box)
            sample_iouN.append(sample_iou.reshape((-1, 1)))
            if ind[0].shape[0] < N / 2 and thre_max >= 0.8:
                if chg_i == 0:
                    Q *= 0.5
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 0.5
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 0.5
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            if ind[0].shape[0] < N / 2 and thre_min <= 0.5:
                if chg_i == 0:
                    Q *= 2
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 2
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 2
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            sample_times += 1
            if sample_times >= 200:  # and cur_n>N/2.0:#100
                # print "Caution: too many loops in sampling"
                #break
                raise OverflowError()
        if cur_n >= N:
            sample_boxN = np.vstack(sample_boxN)[:N, :]
            sample_iouN = np.vstack(sample_iouN)[:N, :]
        else:


            sample_iouN = np.vstack(sample_iouN)
            sample_boxN = np.vstack(sample_boxN)

            diff_n = N - cur_n
            diff_ind = random.sample(range(cur_n), diff_n)  # need to ensure diff_n<cur_n
            sample_boxN = np.vstack([sample_boxN, sample_boxN[diff_ind]])
            sample_iouN = np.vstack([sample_iouN, sample_iouN[diff_ind]])
        return sample_boxN, sample_iouN
    def sample_iou_new(self, gt_box, Q, T, R, N, thre_min=0, thre_max=1):

        sample_boxN = []
        sample_iouN = []
        cur_n = 0
        D = np.array([[self.box_w, 0], [0, self.box_h]])
        QL = D * Q
        # QL = np.linalg.cholesky(Q)
        a = 1.5  # 1.5
        gt_state = utils.bbox_to_states(gt_box, self.area, self.ratio)
        sample_times = 0
        chg_i = 0
        while cur_n < N:
            sample_particles = np.zeros((N, 6), dtype=np.float32)
            QL = D * Q
            sample_particles[:, :2] = gt_state[:, :2] + np.dot(QL,
                                                               np.random.randn(2, N)).transpose()

            if self.mltply:
                dsn = np.random.randn(N) * T
                ds = np.power(a, dsn)
            else:
                ds = 1 + np.random.randn(N) * T
                ds = np.maximum(0.01, ds)  # in case of ds<0
            sample_particles[:, 2] = gt_state[:, 2] * ds

            if self.mltply:
                dr = np.random.randn(N) * R + 1
            else:
                dr = 1 + np.random.randn(N) * R
                dr = np.maximum(0.01, dr)  # in case of dr<0
            sample_particles[:, 3] = gt_state[:, 3] * dr

            # get box
            if self.mltply:
                sample_box = utils.state_to_bbox_m(sample_particles, self.area, self.ratio)
            else:
                sample_box = utils.state_to_bbox(sample_particles, self.area, self.ratio)
            sample_box = utils.restrict_box(sample_box, self.width, self.height)
            # compute iou
            sample_iou = utils.calc_iou(gt_box, sample_box)
            # restrict iou
            ind = np.where((sample_iou >= thre_min) & (sample_iou <= thre_max))
            sample_box = sample_box[ind[0]]
            sample_iou = sample_iou[ind[0]]
            cur_n += sample_box.shape[0]
            sample_boxN.append(sample_box)
            sample_iouN.append(sample_iou.reshape((-1, 1)))
            if ind[0].shape[0] < N / 2 and thre_max >= 0.8:
                if chg_i == 0:
                    Q *= 0.5
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 0.5
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 0.5
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            if ind[0].shape[0] < N / 2 and thre_min <= 0.5:
                if chg_i == 0:
                    Q *= 2
                    chg_i = (chg_i + 1) % 3
                else:
                    if chg_i == 1:
                        T *= 2
                        T = np.minimum(T, 0.5)
                        chg_i = (chg_i + 1) % 3
                    else:
                        R *= 2
                        R = np.minimum(R, 0.5)
                        chg_i = (chg_i + 1) % 3

            sample_times += 1
            if sample_times >= 100:  # and cur_n>N/2.0:#100
                # print "Caution: too many loops in sampling"
                # break
                raise OverflowError()
        if cur_n >= N:
            sample_boxN = np.vstack(sample_boxN)[:N, :]
            sample_iouN = np.vstack(sample_iouN)[:N, :]
        else:
            diff_n = N - cur_n
            sample_iouN = np.vstack(sample_iouN)
            sample_boxN = np.vstack(sample_boxN)
            diff_ind = random.sample(range(cur_n), diff_n)  # need to ensure diff_n<cur_n
            sample_boxN = np.vstack([sample_boxN, sample_boxN[diff_ind]])
            sample_iouN = np.vstack([sample_iouN, sample_iouN[diff_ind]])
        return sample_boxN, sample_iouN

    def run(self, conf):
        self.predict_particles()
        # other process to get conf
        self.update_particles(conf)
        if self.neff() < len(self.particles) / 2:
            self.resample()
        self.estimate()
        return self.cur_pos

    def sample_xy(self, pred_box1, num, thre_max=0.5, thre_min=0.0):
        # only perform xy shift
        # x alone
        pred_box = pred_box1.squeeze()
        alone = 1 / 3.0
        both = 1 - np.sqrt(2 / 3.0)
        pred_box_w = pred_box[2] - pred_box[0]
        pred_box_h = pred_box[3] - pred_box[1]
        pred_box_cx = pred_box[0] + pred_box_w / 2.0
        pred_box_cy = pred_box[1] + pred_box_h / 2.0
        samples = np.ones((num, 4), dtype=np.float32)
        samples_c = np.ones((num, 2), dtype=np.float32)
        samples_c[:, 0] *= pred_box_cx
        samples_c[:, 1] *= pred_box_cy
        num1 = num / 2
        # +-dx,+-dy
        dx = 0.25 + np.random.chisquare(3, num1) * 0.1
        samples_c[:num1 / 4, 0] += (pred_box_w * dx[:num1 / 4])  # dx
        samples_c[num1 / 4:num1 / 2, 0] -= (pred_box_w * dx[num1 / 4:num1 / 2])  # -dx
        samples_c[num1 / 2:3 * num1 / 4, 1] += (pred_box_h * dx[num1 / 2:num1 * 3 / 4])  # dy
        samples_c[3 * num1 / 4:num1, 1] -= (pred_box_h * dx[3 * num1 / 4:])  # -dy
        # |dx2|=|dy2|
        dx2 = 0.15 + np.random.chisquare(3, num) * 0.03
        # dx,dy
        samples_c[num1:(num1 + num1 / 4), 0] += (pred_box_w * dx2[:num1 / 4])
        samples_c[num1:(num1 + num1 / 4), 1] += (pred_box_h * (dx2[num1 / 4:num1 / 2]))
        # -dx,dy
        samples_c[(num1 + num1 / 4):(num1 + num1 / 2), 0] -= (pred_box_w * dx2[num1 / 2:num1 * 3 / 4])
        samples_c[(num1 + num1 / 4):(num1 + num1 / 2), 1] += (pred_box_h * (dx2[3 * num1 / 4:num1]))
        # dx,-dy
        samples_c[(num1 + num1 / 2):(num1 + 3 * num1 / 4), 0] += (pred_box_w * dx2[num1:(num1 + num1 / 4)])
        samples_c[(num1 + num1 / 2):(num1 + 3 * num1 / 4), 1] -= (
                    pred_box_h * (dx2[(num1 + num1 / 4):(num1 + num1 / 2)]))
        # -dx,-dy
        samples_c[(num1 + 3 * num1 / 4):, 0] -= (pred_box_w * dx2[(num1 + num1 / 2):(num1 + 3 * num1 / 4)])
        samples_c[(num1 + 3 * num1 / 4):, 1] -= (
                pred_box_h * (dx2[(num1 + 3 * num1 / 4):]))
        # (x_c,y_c)-->(x1,y1,x2,y2)
        samples[:, 0] = samples_c[:, 0] - pred_box_w / 2.0
        samples[:, 1] = samples_c[:, 1] - pred_box_h / 2.0
        samples[:, 2] = samples_c[:, 0] + pred_box_w / 2.0
        samples[:, 3] = samples_c[:, 1] + pred_box_h / 2.0
        samples = utils.restrict_box(samples, self.width, self.height)
        iou = utils.calc_iou(pred_box, samples)
        iou = iou[:, np.newaxis]
        return samples, iou

    def sample_area(self, pred_box1, num, thre_max=0.5, thre_min=0):
        # sample area
        pred_box = pred_box1.squeeze()
        pred_box_w = pred_box[2] - pred_box[0]
        pred_box_h = pred_box[3] - pred_box[1]
        pred_box_cx = pred_box[0] + pred_box_w / 2.0
        pred_box_cy = pred_box[1] + pred_box_h / 2.0
        samples = np.ones((num, 4), dtype=np.float32)
        samples_wh = np.ones((num, 2), dtype=np.float32)
        # samples_wh[:,0]*=pred_box_w
        # samples_wh[:,1]*=pred_box_h
        cur_n = 0
        # s:(0.3,0.7664)
        s = 0.23 + np.random.chisquare(3, num) * 0.09
        s = 1 - s
        ind = np.where(s > 0.3)[0]
        s = s[ind[:num / 2]]
        samples_wh[:num / 2, 0] = pred_box_w * s
        samples_wh[:num / 2, 1] = pred_box_h * s
        # s:(1.3258,2)
        s = 1.325 + np.random.chisquare(3, num) * 0.125
        ind = np.where(s <= 2)[0]
        s = s[ind[:num / 2]]
        samples_wh[num / 2:, 0] = pred_box_w * s
        samples_wh[num / 2:, 1] = pred_box_h * s
        # (x_c,y_c)-->(x1,y1,x2,y2)
        samples[:, 0] = pred_box_cx - samples_wh[:, 0] / 2.0
        samples[:, 1] = pred_box_cy - samples_wh[:, 1] / 2.0
        samples[:, 2] = pred_box_cx + samples_wh[:, 0] / 2.0
        samples[:, 3] = pred_box_cy + samples_wh[:, 1] / 2.0
        samples = utils.restrict_box(samples, self.width, self.height)
        iou = utils.calc_iou(pred_box, samples)
        iou = iou[:, np.newaxis]
        return samples, iou

    def sample_init(self, gt_box, num_true, num_false):
        boxes_train = []
        # boxes_train_neg=[]
        iou_train = []
        try:
            # Q=[[1,0],[0,1]] #for pixel wise
            Q = 0.05  # box_w,box_h
            sample_box_true, sample_iou_true = self.sample_iou(gt_box, Q, 0.01, 0.01, num_true, 0.8,
                                                               1.0)
        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_true[:10]
        # print sample_box_true.shape[0]
        # print sample_iou_true[:10]
        # print "average iou: ", np.mean(sample_iou_true)
        boxes_train.append(sample_box_true)
        iou_train.append(sample_iou_true)

        try:
            # Q=[[36,0],[0,36]]#for pixel wise
            # Q=0.1
            sample_box_false, sample_iou_false = self.sample_xy(gt_box, num_false / 2)

        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_false[:10]
        # print sample_box_false.shape[0]
        # print sample_iou_false[:10]
        # print "average iou: ", np.mean(sample_iou_false)
        boxes_train.append(sample_box_false)
        iou_train.append(sample_iou_false)

        try:
            # Q=[[36,0],[0,36]]#for pixel wise
            # Q=0.1
            sample_box_false, sample_iou_false = self.sample_iou(gt_box, Q, 0.2, 0.2, num_false / 2, 0, 0.5)

        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_false[:10]
        # print sample_box_false.shape[0]
        # print sample_iou_false[:10]
        # print "average iou: ", np.mean(sample_iou_false)
        boxes_train.append(sample_box_false)
        iou_train.append(sample_iou_false)

        boxes_train = np.vstack(boxes_train)

        iou_train = np.vstack(iou_train)
        y_train_true = np.ones((num_true,))
        y_train_false = np.zeros((num_false,))

        y_train = np.hstack([y_train_true, y_train_false])
        return boxes_train, iou_train, y_train

    def sample_new(self, pred_box, new_true, new_false):
        boxes_train = []

        iou_train = []
        Q = 0.02
        try:
            sample_box_true, sample_iou_true = self.sample_iou(pred_box, Q, 0.01, 0.01, new_true,
                                                               0.85,
                                                               1.0)
        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_true[:10]
        # print sample_box_true.shape[0]
        # print sample_iou_true[:10]
        # print "average iou: ", np.mean(sample_iou_true)
        boxes_train.append(sample_box_true)
        iou_train.append(sample_iou_true)
        try:
            Q = 0.15
            # sample_box_false, sample_iou_false = filter.sample_iou(pred_box, Q, 0.01, 0.01,
            #                                                       new_false / 2, 0,0.5)
            sample_box_false, sample_iou_false = self.sample_xy(pred_box, new_false / 2)
        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_false[:10]
        # print sample_box_false.shape[0]
        # print sample_iou_false[:10]
        # print "average iou: ", np.mean(sample_iou_false)
        boxes_train.append(sample_box_false)
        iou_train.append(sample_iou_false)

        try:
            Q = 0.05
            sample_box_false, sample_iou_false = self.sample_iou(pred_box, Q, 0.2, 0.2,
                                                                 new_false / 2, 0, 0.5)
            # sample_box_false, sample_iou_false = self.sample_xy(pred_box, new_false/2)
        except OverflowError as e:
            print "too many loops in sample."
        # print sample_box_false[:10]
        # print sample_box_false.shape[0]
        # print sample_iou_false[:10]
        # print "average iou: ", np.mean(sample_iou_false)
        boxes_train.append(sample_box_false)
        iou_train.append(sample_iou_false)

        boxes_train = np.vstack(boxes_train)

        iou_train = np.vstack(iou_train)
        y_train_true = np.ones((new_true,))
        y_train_false = np.zeros((new_false,))
        y_train = np.hstack([y_train_true, y_train_false])
        return boxes_train, iou_train, y_train


if __name__ == '__main__':
    root_path = "/home/ccjiang/Documents/py-faster-rcnn/caffe-fast-rcnn/examples/tracker/"

    # vis = args.vis
    # debug = args.debug
    # save = args.save

    dataset_path = "/data/OTB50"
    sequence = "Couple"
    for t in os.walk(os.path.join(dataset_path, sequence, "img")):
        if t[0] == os.path.join(dataset_path, sequence, "img"):
            nFrame = len(t[2])
            print "Total frames are: ", nFrame

    gt_path = os.path.join(dataset_path, sequence, sequence, "groundtruth_rect.txt")

    try:
        gt_str = np.loadtxt(gt_path, dtype=str, delimiter='\n')
        # gt_boxes = map(lambda x: map(float, x.split(',')), gt_str)
        gt_boxes = map(lambda x: map(float, re.split("[, \t]", x)), gt_str)
        gt_boxes = np.array(gt_boxes, dtype=float)
        gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:]  # (x1,y1,w,h) -> (x1,y1,x2,y2)
        # gt_box=gt_boxes[id-1]
    except IOError:
        print "Fail to open ", gt_path

    for id in np.arange(0, 100):
        frame_name = "img/%04d.jpg" % (id + 1)
        print "Start processing: %s" % frame_name
        frame_path = os.path.join(dataset_path, sequence, sequence, frame_name)
        frame_data = cv2.imread(frame_path)  # (432,576,3), in [0,1]
        gt_box = gt_boxes[id]

        if id == 0:
            h, w, c = frame_data.shape
            frame_shape = [c, w, h]
            area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            ratio = (gt_box[2] - gt_box[0]) / (gt_box[3] - gt_box[1])  # ratio=w/h
            # set up net.blobs['im_info']
            print "Image Size: ", w, h

            filter = PFfilter(utils.bbox_to_states(gt_box, area, ratio), area, ratio)
            filter.create_particles()
        # in case of ground_truth box [328,-5,360,30]
        gt_box[0] = np.minimum(np.maximum(0, gt_box[0]), w)
        gt_box[2] = np.minimum(np.maximum(0, gt_box[2]), w)
        gt_box[1] = np.minimum(np.maximum(0, gt_box[1]), h)
        gt_box[3] = np.minimum(np.maximum(0, gt_box[3]), h)

        cv2.rectangle(frame_data, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 2, 1)
        show_particles = 0
        if show_particles:
            for i in range(filter.num_particles):
                cx = filter.particles[i, 0]
                cy = filter.particles[i, 1]
                cv2.circle(frame_data, (int(cx), int(cy)), 1, (0, 255, 0), thickness=1)
        filter.predict_particles()
        # np.save('particles.npy',filter.particles)
        filter.restrict_particles(w, h)
        if show_particles:
            for i in range(filter.num_particles):
                cx = filter.particles[i, 0]
                cy = filter.particles[i, 1]
                cv2.circle(frame_data, (int(cx), int(cy)), 1, (0, 0, 255), thickness=1)
        # compute conf
        conf = np.zeros(filter.weights.shape)
        # np.save('particles.npy',filter.particles)
        pred_boxes = utils.state_to_bbox(filter.particles, area, ratio)

        # print 'pred_boxes: ',pred_boxes
        # for i in range(conf.shape[0]):
        #    pred_box=pred_boxes[i,:]
        # print "pred_box is: ",pred_box
        # conf[i]=np.dot(gt_box,pred_box)/np.linalg.norm(gt_box,ord=2)
        #    conf[i]=np.dot(gt_box,pred_box)/np.sum(np.square(gt_box))
        conf = utils.calc_iou(gt_box, pred_boxes)
        # print 'conf is: ',conf
        filter.update_particles(conf)
        if filter.neff() < len(filter.particles):  # 1/2
            filter.resample()
        pred_state = filter.estimate()
        pred_box = utils.state_to_bbox(pred_state.reshape((-1, 6)), area, ratio)
        print 'ground truth bbox is: ', gt_box
        print "pred_box is: ", pred_box
        iou = utils.calc_iou(gt_box, pred_box)
        print 'iou is: ', iou
        # (B,G,R)
        show_frame = True
        cv2.circle(frame_data, (int(filter.cur_pos[0, 0]), int(filter.cur_pos[0, 1])), 2, (0, 0, 255), thickness=1)
        if show_frame:
            cv2.imshow(sequence, frame_data)
            c = cv2.waitKey()
            print 'You press: ', chr(c)
            if chr(c) == 'c':
                cv2.destroyWindow(sequence)
                break





