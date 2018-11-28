import cv2
import PFfilter
import os
import time
import random
from collections import defaultdict
import pickle
import numpy as np
import utils
import re
import argparse
import VGGnet
import caffe
import sklearn
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def write_res(pred_hist,f):
    '''

    :param pred_hist: (x1,y1,w,h)
    :param f:
    :return:
    '''
    for i in np.arange(pred_hist.shape[0]):
        f.write("%f\t%f\t%f\t%f\n"%(pred_hist[i,0],pred_hist[i,1],pred_hist[i,2],pred_hist[i,3]))
    return f
def DataAugment(img,gt_box,flip=True):
    '''

    :param img:
    :param gt_box:(xc,yc,w,h)
    :param Nout:
    :return:
    '''
    h,w,c=img.shape
    #scale
    s1=np.random.rand(1)*0.2#0.2
    s2=np.random.rand(1)*0.2#0.2
    print "s1: %f"%s1
    print "s2: %f"%s2
    gt_box=np.squeeze(gt_box)
    #bw=gt_box[2]
    #bh=gt_box[3]
    cx=gt_box[0]+gt_box[2]/2.0
    cy=gt_box[1]+gt_box[3]/2.0
    bw,bh=gt_box[2:]

    #new_bw=(1+s)*bw
    #new_bh=(1+s)*bh
    ctxl_w=cx-bw/2.0
    ctxl_h=cy-bh/2.0
    ctxr_w=w-cx-bw/2.0
    ctxr_h=h-cy-bh/2.0
    img1=img[int(s1*ctxl_h):int(h-s2*ctxr_h),int(s1*ctxl_w):int(w-s2*ctxr_w)]
    nh,nw,c=img1.shape
    gt_box1=np.zeros_like(gt_box)
    np.copyto(gt_box1,gt_box)
    gt_box1[0]-=s1*ctxl_w
    gt_box1[1]-=s1*ctxl_h

    #img1 = cv2.rectangle(img1, (int(gt_box1[0]), int(gt_box1[1])),
    #                     (int(gt_box1[0] + gt_box1[2]), int(gt_box1[1] + gt_box1[3])), (0, 0, 255), 3)

    cx_img0=w/2.0
    cy_img0=h/2.0
    cx_img1=nw/2.0
    cy_img1=nh/2.0
    sx=w/float(nw)
    sy=h/float(nh)
    img1 = cv2.resize(img1,(w,h))
    print img1.shape
    gt_box1[0]=cx_img0+sx*(gt_box1[0]-cx_img1)
    gt_box1[1]=cy_img0+sy*(gt_box1[1]-cy_img1)
    gt_box1[2]*=sx
    gt_box1[3]*=sy
    if flip == True:
        #flip horizontally
        img2v=img[:,::-1,:]
        img2v=img2v.copy()
        gt_box2v=np.zeros_like(gt_box)
        np.copyto(gt_box2v,gt_box)
        gt_box2v[0]=w-(gt_box2v[0]+gt_box2v[2])

    #rotation
    theta = np.random.rand(1)*10-5
    #theta = np.random.rand(1)*15-7.5
    print "theta: %f"%theta
    rot_mat=cv2.getRotationMatrix2D((w/2.0,h/2.0),theta,1.0)
    imgr=cv2.warpAffine(img,rot_mat,(w,h))

    gt_boxr=np.zeros_like(gt_box)
    np.copyto(gt_boxr,gt_box)
    points=np.ones((3,4))
    points[0,0]=gt_box[0]
    points[1,0]=gt_box[1]
    points[0,1]=gt_box[0]+gt_box[2]
    points[1,1]=gt_box[1]
    points[0,2]=gt_box[0]+gt_box[2]
    points[1,2]=gt_box[1]+gt_box[3]
    points[0,3]=gt_box[0]
    points[1,3]=gt_box[1]+gt_box[3]

    points_new=np.dot(rot_mat,points)
    xl=np.min(points,axis=1)[0]
    yl=np.min(points,axis=1)[1]
    xr=np.max(points,axis=1)[0]
    yr=np.max(points,axis=1)[1]
    gt_boxr[0]=xl
    gt_boxr[1]=yl
    gt_boxr[2]=xr-xl
    gt_boxr[3]=yr-yl
    if flip == True:
        return img1,gt_box1,img2v,gt_box2v,imgr,gt_boxr
    else:
        return img1,gt_box1,imgr,gt_boxr
def featmap_pca2(features,ncompnents=128):
    # features: (256,h,w)
    # feat=np.transpose(features,(1,2,0))#(h,w,c)
    feat = features.reshape((features.shape[0], -1))  # (c,h*w)
    feat = feat.transpose()
    # feat = sklearn.preprocessing.scale(feat, axis=0, with_mean=True, with_std=True)
    scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    scaler.fit(feat)
    feat = scaler.transform(feat)
    pca_1 = PCA(n_components=ncompnents)
    pca_1.fit(feat)
    feat2 = pca_1.transform(feat)
    feat2 = feat2.transpose()
    feat2 = feat2.reshape((ncompnents, features.shape[1], features.shape[2]))

    return pca_1, scaler


def feat_transformpca(pca, scaler, features):
    # input features:(N,256,h,w)
    # output features: (N,128,h,w)
    ncomp = 128#128
    channels_in = 256#512
    features=features.reshape((-1,channels_in,7,7))
    ori_shape = features.shape
    N=ori_shape[0]
    channels=ori_shape[1]
    h=ori_shape[2]
    w=ori_shape[3]
    features = features.reshape((N, channels,h*w))#(N,256,h*w)
    features = features.transpose((0,2,1))#(N,h*w,256)
    features=features.reshape((-1,channels))#(N*h*w,256)
    features = scaler.transform(features)

    features = pca.transform(features)#(N*h*w,128)
    features=features.reshape((N,h,w,ncomp))
    features = features.transpose((0,3,1,2)) # (N,128,h,w)
    features=features.reshape((N,-1))

    return features


def nms_box(preds, confm, feat, conf):
    # preds:(N,4),(x1,y1,x2,y2)
    # N=preds.shape[0]
    res_box = []
    res_feat = []
    res_conf = []
    thresh = 0.5  # 0.7
    t = []
    while preds.shape[0] > 0:
        ind = np.argsort(-confm.squeeze()).tolist()

        i = ind[0]
        cur_box = preds[i]
        cur_iou = utils.calc_iou(cur_box, preds)
        i_spa = np.where(cur_iou < thresh)[0]
        res_box.append(cur_box)
        res_feat.append(feat[i])
        res_conf.append(conf[i, :])
        t.append(preds.shape[0] - i_spa.shape[0])
        preds = preds[i_spa]
        confm = confm[i_spa]
        feat = feat[i_spa]
        conf = conf[i_spa]
    return np.array(res_box), np.array(res_feat), np.array(res_conf), np.array(t)

def nms_pred(pred,preds, vpca3,conf):
    # preds:(N,4),(x1,y1,x2,y2)
    # N=preds.shape[0]

    thresh = 0.3  # 0.7
    cur_iou = utils.calc_iou(pred, preds)
    i_spa = np.where(cur_iou < thresh)[0]
    i_spa2 = np.where(conf > 0.4)[0]
    print "IOU: ",i_spa
    print "Conf: ",i_spa2
    ind = np.array(list(set(i_spa).intersection(set(i_spa2))))
    print ind
    if ind.shape[0] > 0:
        res_box = preds[ind,:]
        res_vpca3 = vpca3[ind,:]
        return True,res_vpca3
    else:
        return False,np.zeros((1,4))


def calc_gauss_dist(pred_box, box):
    pred_box = pred_box.squeeze()
    pred_box = pred_box[np.newaxis, :]
    box = box.squeeze()
    sigm = 10.0
    d = (pred_box - box) * (pred_box - box) / 2.0
    s = np.exp(-d / sigm / sigm)
    return s


def calc_entroy(feats, clf):
    eps = 1e-6
    N = feats.shape[0]
    # conf_t = f.predict_proba(v_pca3)[:, 1]
    s = clf.predict_proba(feats)[:, 1]

    # softmax
    d = np.sum(np.exp(s))
    s = np.exp(s) / d
    # calculate entropy
    res = -np.sum(s * np.log10(s + eps))
    return res


def gaussian_label(sz, sigma, im_sz, offset):
    x, y = np.meshgrid(np.arange(-(im_sz[0] / 2), im_sz[0] / 2 + im_sz[0] % 2),
                       np.arange(-(im_sz[1] / 2), im_sz[1] / 2 + im_sz[1] % 2))
    x = x - (offset[0] - im_sz[0] / 2)
    y = y - (offset[1] - im_sz[1] / 2)
    z = np.exp(-0.5 * (np.square(x / float(sz[0] * sigma)) + np.square(y / float(sz[1] * sigma))))
    return z


def main(args):
    vis = args.vis
    debug = args.debug
    save = args.save
    nparticles = args.particles
    root_path = '/home/ccjiang/Documents/py-faster-rcnn/caffe-fast-rcnn/examples/tracker/'
    dataset_path = args.dataset  # "/data/OTB100"
    dataset100_seq = ['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board', 'Bolt2', 'Boy', 'Car2',
                      'Car24', 'Coke', 'Coupon', 'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3',
                      'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football1',
                      'Freeman1', 'Freeman3', 'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8',
                      'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik',
                      'Singer1', 'Skater', 'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans',
                      'Twinnings', 'Vase']
    dataset50_seq = ['Basketball',  'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl',
                     'Bolt', 'Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds','David',
                     'Deer', 'Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4', 'Girl',
                     'Human3', 'Human4', 'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor',
                     'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Shaking', 'Singer2', 'Skating1',
                     'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis',
                     'Walking', 'Walking2', 'Woman']
    datafull_seq = dataset100_seq + dataset50_seq
    if "OTB50" in dataset_path:
        data_seq = dataset50_seq
    else:
        data_seq = dataset100_seq

    log_name = 'log_1119.txt'
    log_file = open(log_name, 'w')
    records_success = []  # defaultdict(list)
    records_precision = []  # defaultdict(list)
    records_reinit = defaultdict(list)
    model_def = os.path.join(root_path, args.prototxt)
    model_weight = os.path.join(root_path, args.caffemodel)
    vggnet = VGGnet.VGGnet(model_def, model_weight)

    thre_max_neg = 0.3  # 0.5
    test_times = 1  # 0
    for t in range(test_times):
        print 'Test round: %d' % t
        log_file.write('Test round: %d\n' % t)
        # sequences = ['Fish']
        for sequence in datafull_seq:  # datafull_seq
            if sequence in dataset50_seq:
                dataset_path = "/data/OTB50"
            else:
                dataset_path = "/data/OTB100"
            for t in os.walk(os.path.join(dataset_path, sequence, sequence, "img")):
                if t[0] == os.path.join(dataset_path, sequence, sequence, "img"):
                    nFrame = len(t[2])
                    print 'Processing: %s' % sequence
                    log_file.write('Processing: %s\n' % sequence)
                    print "Total frames are: ", nFrame
                    log_file.write('Total frames are: %d\n' % nFrame)
            gt_path = os.path.join(dataset_path, sequence, sequence, "groundtruth_rect.txt")

            gt_boxes = utils.get_boxes_all(gt_path)

            conf_hist = []
            iou_hist = []
            area_hist = []
            pred_hist = []  # (x1,y1,x2,y2)
            eig_hist = []
            reinit = 0
            nFrame = np.minimum(nFrame, gt_boxes.shape[0])

            id_shift = 0
            init_id = False
            update_recent = False
            for id in np.arange(0, nFrame):
                frame_name = "img/%04d.jpg" % (id + 1)
                # print "Start processing: %s" % frame_name
                frame_path = os.path.join(dataset_path, sequence, sequence, frame_name)
                if os.path.exists(frame_path) == False:
                    id_shift = id_shift + 1
                    continue
                id = id - id_shift

                frame_data = caffe.io.load_image(frame_path)  # (432,576,3), in [0,1]
                gt_box = gt_boxes[id]

                if init_id == False:
                    h, w, c = frame_data.shape
                    frame_shape = [c, w, h]
                    fps = 20
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_writer = cv2.VideoWriter("res_%s.avi"%sequence,fourcc,fps,(w,h))
                    fail_times = 0
                    box_w = gt_box[2] - gt_box[0]
                    box_h = gt_box[3] - gt_box[1]
                    area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    ratio = (gt_box[2] - gt_box[0]) / (gt_box[3] - gt_box[1])  # ratio=w/h
                    # set up net.blobs['im_info']
                    print "Image Size: ", w, h
                    log_file.write('Image Size: %d %d\n' % (w, h))
                    b = gt_box[np.newaxis, :]
                    vggnet.reshape(w=w, h=h, nbox=b.shape[0])
                    features0 = vggnet.get_features("conv3_3", frame_data,
                                                    boxes_raw=b)  # shape:(256,hs,ws),conv3_3,res3b3
                    features0 = np.squeeze(features0)
                    pca_f, scaler_f = featmap_pca2(features0,ncompnents=128)#128

                    box_w = gt_box[2] - gt_box[0]
                    box_h = gt_box[3] - gt_box[1]

                    vggnet.reshape(w=w, h=h, nbox=nparticles)
                    pfilter = PFfilter.PFfilter(utils.bbox_to_states(gt_box, area, ratio), area, ratio, w, h,
                                                nparticles)
                    pfilter.create_particles()
                    pfilter.restrict_particles(w, h)
                    area_hist.append(pfilter.cur_a)
                    pred_hist.append(np.array(gt_box).reshape(1, -1))
                    # pca
                    # test sample_iou
                    num_true = 500
                    num_false = 1000  # 1000
                    #data augument
                    gt_box_otb = gt_box.copy()
                    gt_box_otb[2] -= gt_box_otb[0]
                    gt_box_otb[3] -= gt_box_otb[1]
                    boxes_train = []
                    ids = np.zeros(num_false + num_true)


                    imgs = []
                    for i in np.arange(4):
                        if i == 0:
                            img1,gt1,img2,gt2,img3,gt3 = DataAugment(frame_data,gt_box_otb,True)
                            gt1[2] += gt1[0]
                            gt1[3] += gt1[1]
                            gt2[2] += gt2[0]
                            gt2[3] += gt2[1]
                            gt3[2] += gt3[0]
                            gt3[3] += gt3[1]
                            box_true1, iou_true = pfilter.sample_iou_pred_box(gt1, 0.05, 0.01, 0.01, 20, 0.8, 1.0)
                            box_true2, iou_true = pfilter.sample_iou_pred_box(gt2, 0.05, 0.01, 0.01, 40, 0.8, 1.0)
                            box_true3, iou_true = pfilter.sample_iou_pred_box(gt3, 0.05, 0.01, 0.01, 20, 0.8, 1.0)
                            box_true1[0, ...] = gt1
                            box_true2[0, ...] = gt2
                            box_true3[0, ...] = gt3

                            boxes_train.append(box_true1)
                            boxes_train.append(box_true2)
                            boxes_train.append(box_true3)
                            imgs.append(img1)
                            imgs.append(img2)
                            imgs.append(img3)
                            ids[20:60] = 1
                            ids[60:80] = 2

                        else:
                            img1, gt1, img2, gt2 = DataAugment(frame_data, gt_box_otb,False)
                            gt1[2] += gt1[0]
                            gt1[3] += gt1[1]
                            gt2[2] += gt2[0]
                            gt2[3] += gt2[1]

                            box_true1, iou_true = pfilter.sample_iou_pred_box(gt1, 0.05, 0.01, 0.01, 20, 0.8, 1.0)
                            box_true2, iou_true = pfilter.sample_iou_pred_box(gt2, 0.05, 0.01, 0.01, 20, 0.8, 1.0)

                            box_true1[0, ...] = gt1
                            box_true2[0, ...] = gt2

                            boxes_train.append(box_true1)
                            boxes_train.append(box_true2)
                            imgs.append(img1)
                            imgs.append(img2)
                            cur_i = 80+(i-1)*40
                            ids[cur_i:(cur_i+20)] = 3+(i-1)*2
                            ids[(cur_i+20):(cur_i+40)] = 3+(i-1)*2+1
                    # boxes_train_neg=[]

                    try:
                        # Q=[[1,0],[0,1]] #for pixel wise
                        Q = 0.05  # box_w,box_h,0.05
                        sample_box_true, sample_iou_true = pfilter.sample_iou_pred_box(gt_box, Q, 0.01, 0.01,
                                                                                       num_true-200, 0.8,
                                                                                       1.0)#0.8
                    except OverflowError as e:
                        print "too many loops in sample in Initialize--TRUE."

                    boxes_train.append(sample_box_true)

                    try:
                        # Q=[[36,0],[0,36]]#for pixel wise
                        Q = 0.2  # 0.2
                        sample_box_false, sample_iou_false = pfilter.sample_iou(gt_box, Q, 0.2, 0.01, num_false / 2, 0,
                                                                                thre_max_neg)  # 0.2,0.01
                    except OverflowError as e:
                        print "too many loops in sample in Initialize--FALSE."
                    # print sample_box_false[:10]
                    # print sample_box_false.shape[0]
                    # print sample_iou_false[:10]
                    # print "average iou: ", np.mean(sample_iou_false)
                    boxes_train.append(sample_box_false)

                    try:
                        # Q=[[36,0],[0,36]]#for pixel wise
                        Q = 0.2  # 0.2
                        sample_box_false, sample_iou_false = pfilter.sample_iou(gt_box, Q, 0.01, 0.2, num_false / 2, 0,
                                                                                thre_max_neg)  # 0.01,0.2
                    except OverflowError as e:
                        print "too many loops in sample in Initialize--FALSE."

                    boxes_train.append(sample_box_false)

                    boxes_train = np.vstack(boxes_train)

                    imgs.append(frame_data)
                    imgs = np.stack(imgs,axis=0)#(10,h,w,c)
                    ids[200:] = 9
                    y_train_true = np.ones((num_true,))
                    y_train_false = np.zeros((num_false,))
                    ids_save = np.ones((num_true+num_false))
                    ids_save[num_true:] = 0
                    ids_save[20:60] = 2
                    y_train = np.hstack([y_train_true, y_train_false])

                    # permutation
                    ind_perm = np.random.permutation(range(num_false + num_true))
                    boxes_train = boxes_train[ind_perm, :]
                    ids_save = ids_save[ind_perm]
                    y_train = y_train[ind_perm]
                    ids = ids[ind_perm]
                    ind_pos = np.where(y_train == 1)[0]
                    ind_neg = np.where(y_train == 0)[0]

                    vggnet.reshape(w=w, h=h, nbox=boxes_train.shape[0],batch_size=10)
                    #features = vggnet.get_features_first_raw(frame_data, boxes_raw=boxes_train, id=id)
                    features = vggnet.get_features_first_id(imgs,boxes_raw=boxes_train,id=ids)
                    #features = vggnet.get_features_first_sel(frame_data, boxes_raw=boxes_train, id=id, sel=f_inds)
                    for k, v in features.iteritems():
                        # print k,v.shape
                        if k == 'f3':
                            #pca3, scaler1, nPCA = utils.skl_pca2(v)
                            v = feat_transformpca(pca_f, scaler_f, v)  # (N,128,7,7)
                            pca3,nPCA=utils.skl_pca_noscale(v)
                            #np.save('pca_compns.npy',pca3.components_)
                            v_pca3 = pca3.transform(v)
                            np.save("pca_results/testpca_%s.npy"%sequence,v_pca3)
                            #np.save('labelpca.npy',y_train)
                            np.save("pca_results/label_pca_%s"%sequence,ids_save)
                            pca3_pos = np.zeros((num_true, pca3.n_components_), dtype=np.float32)
                            pca3_neg = np.zeros((num_false, pca3.n_components_), dtype=np.float32)
                            pca3_pos[...] = v_pca3[ind_pos, :]
                            pca3_neg[...] = v_pca3[ind_neg, :]
                            # utils.vis_as_image(v_pca3)
                            # plt.imshow(v_pca3)
                            # plt.title("PCA features")
                            # plt.show()
                            # plt.close()
                            # logistic regression
                            y_weight = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                       classes=np.array([0, 1]),
                                                                                       y=y_train)
                            # print y_weight
                            class_weight = {0: y_weight[0], 1: y_weight[1]}
                            clf3 = linear_model.LogisticRegression(fit_intercept=True, solver='liblinear')
                            clf3.fit(v_pca3, y_train)

                    vis_feature = False
                    if vis_feature:
                        utils.vis_features(features, id)

                    start_time = time.time()
                else:
                    if fail_times >= 5:
                        # reinitialize
                        update_recent = False
                        reinit += 1
                        area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        ratio = (gt_box[2] - gt_box[0]) / (gt_box[3] - gt_box[1])
                        pfilter = PFfilter.PFfilter(utils.bbox_to_states(gt_box, area, ratio), area, ratio, w, h,
                                                    nparticles)
                        # filter.reset(utils.bbox_to_states(gt_box, area, ratio), area, ratio)
                        pfilter.create_particles()
                        pfilter.restrict_particles(w, h)
                        area_hist.append(pfilter.cur_a)
                        pred_box = gt_box
                        pred_hist.append(np.array(pred_box).reshape(1, -1))
                        conf_hist.append(-0.1)
                        boxes_train = []
                        # boxes_train_neg=[]
                        iou_train = []
                        try:
                            # Q=[[1,0],[0,1]] #for pixel wise
                            Q = 0.05  # box_w,box_h,0.05
                            sample_box_true, sample_iou_true = pfilter.sample_iou_pred_box(gt_box, Q, 0.01, 0.01,
                                                                                           num_true, 0.8,
                                                                                           1.0)
                        except OverflowError as e:
                            print "too many loops in sample in Reinitialize--TRUE."

                        boxes_train.append(sample_box_true)
                        iou_train.append(sample_iou_true)
                        try:
                            # Q=[[36,0],[0,36]]#for pixel wise
                            Q = 0.2  # 0.2
                            sample_box_false, sample_iou_false = pfilter.sample_iou(gt_box, Q, 0.01, 0.2, num_false / 2,
                                                                                    0,
                                                                                    thre_max_neg)
                        except OverflowError as e:
                            print "too many loops in sample in Reinitialize--FALSE."

                        boxes_train.append(sample_box_false)
                        iou_train.append(sample_iou_false)
                        try:
                            # Q=[[36,0],[0,36]]#for pixel wise
                            Q = 0.2  # 0.2
                            sample_box_false, sample_iou_false = pfilter.sample_iou(gt_box, Q, 0.2, 0.01, num_false / 2,
                                                                                    0,
                                                                                    thre_max_neg)
                        except OverflowError as e:
                            print "too many loops in sample in Reinitialize--FALSE."
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
                        # permutation
                        ind_perm = np.random.permutation(range(num_false + num_true))
                        boxes_train = boxes_train[ind_perm, :]
                        iou_train = iou_train[ind_perm]
                        y_train = y_train[ind_perm]
                        ind_pos = np.where(y_train == 1)[0]
                        ind_neg = np.where(y_train == 0)[0]

                        vggnet.reshape(w=w, h=h, nbox=boxes_train.shape[0])
                        features = vggnet.get_features_first_raw(frame_data, boxes_raw=boxes_train, id=id)
                        #features=feat_transformpca(pca_f,scaler_f,features)
                        #features = vggnet.get_features_first_sel(frame_data, boxes_raw=boxes_train, id=id, sel=f_inds)
                        for k, v in features.iteritems():
                            # print k, v.shape
                            if k == 'f3':
                                v = feat_transformpca(pca_f, scaler_f, v)  # (N,128,7,7)
                                v_pca3 = pca3.transform(v)

                                pca3_pos[...] = v_pca3[ind_pos, :]
                                pca3_neg[...] = v_pca3[ind_neg, :]
                                clf3 = linear_model.LogisticRegression(fit_intercept=True, solver='liblinear')
                                clf3.fit(v_pca3, y_train)
                                # score3 = clf3.score(v_pca3, y_train)

                                # print 'score3: ', score3
                                # prob=clf3.predict_proba(v_pca3)
                                # print clf3.classes_
                        fail_times = 0
                        continue

                    pfilter.predict_particles(Q=0.2, cr=0.01, ca=0.01)  # 0.2,0.01
                    pfilter.restrict_particles(w, h)
                    area_hist.append(pfilter.cur_a)
                    # compute conf
                    # conf = np.zeros(pfilter.weights.shape)
                    # np.save('particles.npy',filter.particles)
                    pred_boxes = utils.state_to_bbox(pfilter.particles, area, ratio)
                    #add Gaussian regularization
                    if id>1:
                        gauss_sig = 0.5
                        gauss_w = np.exp(-np.square((pfilter.particles[:,0]-pred_state[0])/(gauss_sig*box_w)/2.0)-np.square((pfilter.particles[:,1]-pred_state[1])/(gauss_sig*box_h)/2.0))
                        pfilter.update_particles(gauss_w)
                        print gauss_w
                    vggnet.reshape(w, h, pfilter.num_particles)
                    features = vggnet.get_features_first_raw(frame_data, boxes_raw=pred_boxes, id=id)
                    #features=feat_transformpca(pca_f,scaler_f,features)
                    #features = vggnet.get_features_first_sel(frame_data, boxes_raw=pred_boxes, id=id, sel=f_inds)
                    for k, v in features.iteritems():
                        # print k,v.shape
                        if k == 'f3':
                            v = feat_transformpca(pca_f, scaler_f, v)  # (N,128,7,7)
                            v_pca3 = pca3.transform(v)
                            conf = clf3.predict_proba(v_pca3)[:, 1]

                    # process preds to find out pred_box in terms of confm
                    conf = np.array(conf)

                    conf_max = np.max(conf)
                    conf_min = np.min(conf)
                    pfilter.update_particles(conf)
                    # do resample first or estimate first?
                    # filter.resample()  # always resample
                    pred_state, s_particles, r_particles = pfilter.estimate(k=10)
                    pfilter.resample()
                    pred_box = utils.state_to_bbox(pred_state.reshape((-1, 6)), area, ratio)

                    hard,hard_negv = nms_pred(pred_box,pred_boxes,v_pca3,conf)
                    if hard:
                        hard_negvN = hard_negv.shape[0]
                    else:
                        hard_negvN = 0

                    avg_pos = np.mean(pfilter.particles[:, :2], axis=0, keepdims=True)
                    # avg_pos[:,0]/=w
                    # avg_pos[:,1]/=h
                    ptls_avg = (pfilter.particles[:, :2] - avg_pos) / np.array([[box_w, box_h]])
                    cov_particles = np.dot(ptls_avg.T, ptls_avg) / pfilter.particles.shape[
                        0]

                    eigval, eigvec = np.linalg.eig(cov_particles)
                    max_val = eigval[0]
                    eig_hist.append(max_val)
                    print 'Max eigvalue: %f' % max_val

                    # print 'conf is: ',conf
                    if conf_max > 0.5:  # 0.8
                        fail_times = 0
                        update_recent = False
                    else:
                        fail_times += 1

                    show_sr = False
                    if show_sr:
                        count, xedge, yedge, tmp_im = plt.hist2d(s_particles, r_particles, bins=10,
                                                                 weights=pfilter.weights.squeeze(), cmap=plt.cm.gray)
                        top3 = np.argsort(-count, axis=None)[:3]
                        row_ind = top3[:] / count.shape[1]
                        col_ind = top3[:] % count.shape[0]

                        plt.show()
                    print pred_box
                    iou = utils.calc_iou(gt_box, pred_box)
                    # print 'iou is: ', iou
                    pred_hist.append(pred_box)
                    conf_hist.append(conf_max)
                    iou_hist.append(iou)

                    if conf_max >= 0.7:  # 0.5
                        # update pca3_pos and pca3_neg
                        new_true = 100  # 100
                        new_false = 400  # 200
                        boxes_train = []

                        iou_train = []
                        Q = 0.05  # 0.02
                        try:
                            sample_box_true, sample_iou_true = pfilter.sample_iou_pred_box(pred_box, Q, 0.01, 0.01,
                                                                                           new_true,
                                                                                           0.85,
                                                                                           1.0)
                        except OverflowError as e:
                            print "too many loops in sample in Update--TRUE."
                        # print sample_box_true[:10]
                        # print sample_box_true.shape[0]
                        # print sample_iou_true[:10]
                        # print "average iou: ", np.mean(sample_iou_true)
                        boxes_train.append(sample_box_true)

                        iou_train.append(sample_iou_true)
                        # part_iou=utils.calc_iou(pred_box,pred_boxes)

                        # ind_iou=np.where(part_iou<0.3)[0]

                        # ind_n=np.minimum(new_false/2,ind_iou.shape[0])
                        # boxes_train.append(pred_boxes[ind_iou[:ind_n],:])
                        # iou_train.append(part_iou[ind_iou])
                        new_false_left = new_false - hard_negvN  # -ind_n
                        try:
                            Q = 0.2  # 0.2
                            sample_box_false, sample_iou_false = pfilter.sample_iou_pred_box(pred_box, Q, 0.2, 0.01,
                                                                                             (new_false_left + 1) / 2,
                                                                                             0, thre_max_neg)
                        except OverflowError as e:
                            print "too many loops in sample in Update--FALSE."
                        # print sample_box_false[:10]
                        # print sample_box_false.shape[0]
                        # print sample_iou_false[:10]
                        # print "average iou: ", np.mean(sample_iou_false)
                        boxes_train.append(sample_box_false)
                        iou_train.append(sample_iou_false)
                        try:
                            Q = 0.2  # 0.2
                            sample_box_false, sample_iou_false = pfilter.sample_iou_pred_box(pred_box, Q, 0.01, 0.2,
                                                                                             new_false_left / 2, 0,
                                                                                             thre_max_neg)
                        except OverflowError as e:
                            print "too many loops in sample in Update--FALSE."
                        boxes_train.append(sample_box_false)
                        iou_train.append(sample_iou_false)

                        boxes_train = np.vstack(boxes_train)

                        # iou_train = np.vstack(iou_train)




                        vggnet.reshape(w=w, h=h, nbox=boxes_train.shape[0])
                        features = vggnet.get_features_second_raw(boxes_raw=boxes_train, id=id)
                        #features = feat_transformpca(pca_f,scaler_f,features)
                        #features = vggnet.get_features_second_sel(boxes_raw=boxes_train, id=id, sel=f_inds)
                        for k, v in features.iteritems():
                            # print k, v.shape
                            if k == 'f3':
                                v = feat_transformpca(pca_f, scaler_f, v)  # (N,128,7,7)
                                v_pca3 = pca3.transform(v)
                                if hard:
                                    v_pca3 = np.vstack([v_pca3,hard_negv])
                                y_train_true = np.ones((new_true,))
                                y_train_false = np.zeros((new_false,))
                                y_train = np.hstack([y_train_true, y_train_false])
                                # permutation
                                ind_perm = np.random.permutation(range(new_false + new_true))
                                #boxes_train = boxes_train[ind_perm, :]
                                v_pca3 = v_pca3[ind_perm,:]
                                y_train = y_train[ind_perm]
                                new_y = np.zeros(y_train.shape)
                                new_y[...] = y_train
                                ind_pos = np.where(y_train == 1)[0]
                                ind_neg = np.where(y_train == 0)[0]

                                # random substitude
                                pca3_cur_pos = v_pca3[ind_pos, :]
                                pca3_cur_neg = v_pca3[ind_neg, :]
                                to_subst = random.sample(range(num_true), new_true)
                                pca3_pos[to_subst, :] = pca3_cur_pos
                                to_subst = random.sample(range(num_false), new_false)
                                pca3_neg[to_subst, :] = pca3_cur_neg


                    if conf_max < 1 and fail_times >= 2 and update_recent==False:
                        # if id%10==0:
                        update_recent = True
                        pca3_train = np.vstack([pca3_pos, pca3_neg])

                        y_train_true = np.ones((num_true,))
                        y_train_false = np.zeros((num_false,))
                        y_train = np.hstack([y_train_true, y_train_false])

                        # permutation
                        ind_perm = np.random.permutation(range(num_false + num_true))
                        pca3_train = pca3_train[ind_perm, :]

                        y_train = y_train[ind_perm]

                        # logistic regression
                        clf3 = linear_model.LogisticRegression(fit_intercept=True, solver='liblinear')
                        clf3.fit(pca3_train, y_train)

                        # print 'score is: ',clf3.score(pca3_train,y_train)

                # (B,G,R)
                frame_data_cv = frame_data * 255  # [0,1]-->[0,255]
                frame_data_cv = frame_data_cv[:, :, ::-1]  # RGB->BGR
                frame_data_cv = frame_data_cv.astype('uint8')
                cv2.rectangle(frame_data_cv, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])),
                              (255, 0, 0), 2, 1)
                if id > 0 and init_id == True:
                    cv2.rectangle(frame_data_cv, (int(pred_box[0, 0]), int(pred_box[0, 1])),
                                  (int(pred_box[0, 2]), int(pred_box[0, 3])),
                                  (0, 255, 0), 2, 1)
                if init_id == False:
                    init_id = True
                show_particles = False
                if show_particles:
                    for i in range(filter.num_particles):
                        cx = pfilter.particles[i, 0]
                        cy = pfilter.particles[i, 1]
                        cv2.circle(frame_data_cv, (int(cx), int(cy)), 1, (0, 0, 255), thickness=1)
                show_box = False
                if show_box:
                    n = 0
                    for i in ind_pos:
                        if n % 5 == 0:
                            cv2.rectangle(frame_data_cv, (int(boxes_train[i, 0]), int(boxes_train[i, 1])),
                                          (int(boxes_train[i, 2]), int(boxes_train[i, 3])), (0, 0, 255), 2, 1)
                        n += 1
                    n = 0

                show_particles_init = False
                if show_particles_init:
                    for i in range(filter.num_particles):
                        cx = pfilter.particles[i, 0]
                        cy = pfilter.particles[i, 1]
                        cv2.circle(frame_data_cv, (int(cx), int(cy)), 1, (0, 255, 0), thickness=1)
                show_frame = False
                cv2.circle(frame_data_cv, (int(pfilter.cur_c[0]), int(pfilter.cur_c[1])), 2, (0, 0, 255), thickness=1)
                if show_frame:
                    cv2.imshow(sequence, frame_data_cv)
                    c = cv2.waitKey(1)

                    if c != -1:
                        cv2.destroyWindow(sequence)

                        break
                else:
                    video_writer.write(frame_data_cv)
            end_time = time.time()
            video_writer.release()
            print "Average FPS: %f" % (nFrame / (end_time - start_time))
            log_file.write("Average FPS: %f\n" % (nFrame / (end_time - start_time)))
            conf_hist = np.array(conf_hist)
            iou_hist = np.array(iou_hist)
            area_hist = np.array(area_hist)
            pred_hist = np.vstack(pred_hist)
            precisions, auc_pre = utils.calc_prec(gt_boxes, pred_hist)

            suc, auc_iou = utils.calc_success(iou_hist)
            records_precision.append(precisions * nFrame)
            records_success.append(suc * nFrame)

            print 'Precision @20 is: %f' % precisions[19]
            print 'Auc of Precision is: %f' % auc_pre
            print 'Auc of Success is: %f' % auc_iou
            print 'Reinit times: %d' % reinit
            log_file.write("Precision @20 is: %f\n" % precisions[19])
            log_file.write('Auc of Precision is: %f\n' % auc_pre)
            log_file.write('Auc of Success is: %f\n' % auc_iou)
            log_file.write('Reinit times: %d\n' % reinit)
            #log_file.write('Selected feature maps: %d\n' % f_inds.shape[0])
            log_file.write('PCA components: %d\n' % nPCA)
            res_f = open('results11/%s.txt'%sequence,'w')
            pred_hist[:,2:] = pred_hist[:,2:] - pred_hist[:,:2]
            res_f = write_res(pred_hist,res_f)
            res_f.close()
    log_file.close()
    pkl = open('results_1119.pkl', 'w')
    pickle.dump([records_precision, records_success], pkl)
    pkl.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--prototxt", default='vgg16_align_conv3.prototxt', type=str)
    parser.add_argument("--caffemodel", default='VGG_ILSVRC_16_layers.caffemodel', type=str)
    parser.add_argument("--sequence", default="Car4", type=str)
    parser.add_argument("--classifier", default='svm', type=str)
    parser.add_argument("--particles", default=100, type=int)
    parser.add_argument("--dataset", default="/data/OTB50", type=str)
    args = parser.parse_args()
    main(args)
