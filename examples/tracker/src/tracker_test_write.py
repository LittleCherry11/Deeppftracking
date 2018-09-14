import cv2
import PFfilter
import  os
import time
import random
import numpy as np
import utils
import re
import argparse
import VGGnet
import caffe
import sklearn
from sklearn import linear_model

import matplotlib.pyplot as plt
def main(args):


    vis = args.vis
    debug = args.debug
    save = args.save
    nparticles=args.particles


    root_path='/home/ccjiang/Documents/caffe-fast-rcnn/examples/tracker/'
    dataset_path = "/data/OTB100"
    sequence = args.sequence
    model_def = os.path.join(root_path, args.prototxt)
    model_weight = os.path.join(root_path, args.caffemodel)
    for t in os.walk(os.path.join(dataset_path, sequence,sequence, "img")):
        if t[0] == os.path.join(dataset_path, sequence,sequence, "img"):
            nFrame = len(t[2])
            print "Total frames are: ", nFrame

    gt_path = os.path.join(dataset_path, sequence,sequence,"groundtruth_rect.txt")

    gt_boxes=utils.get_boxes_all(gt_path)

    vggnet=VGGnet.VGGnet(model_def,model_weight)

    thre_min_neg=0.0
    thre_max_neg=0.3#0.4 #0.5

    thre_min_pos=0.8
    thre_max_pos=1.0

    conf_hist=[]
    iou_hist=[]
    area_hist=[]
    eig_hist=[]
    pred_hist=[]#(x1,y1,x2,y2)
    reinit=0
    nFrame=np.minimum(nFrame,gt_boxes.shape[0])
    init_id=False
    id_shift=0
    for id in np.arange(0,nFrame):#nFrame
        frame_name = "img/%04d.jpg" % (id + 1)
        print "Start processing: %s" % frame_name
        frame_path = os.path.join(dataset_path, sequence, sequence,frame_name)
        if os.path.exists(frame_path)==False:
           id_shift=id_shift+1
           continue
        id=id-id_shift
        frame_data = caffe.io.load_image(frame_path)  # (432,576,3), in [0,1]
        gt_box=gt_boxes[id]


        if init_id == False:#id==0
            #init_id=True
            h,w,c = frame_data.shape
            frame_shape = [c, w, h]
            fps = 20
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            video_writer = cv2.VideoWriter("res_%s.avi"%sequence, fourcc, fps,(w,h))
            fail_times=0
            area=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
            ratio=(gt_box[2]-gt_box[0])/(gt_box[3]-gt_box[1]) #ratio=w/h
            # set up net.blobs['im_info']
            print "Image Size: ", w, h
            vggnet.reshape(w=w,h=h,nbox=nparticles)
            filter=PFfilter.PFfilter(utils.bbox_to_states(gt_box, area, ratio), area, ratio, w, h, nparticles)
            filter.create_particles()
            filter.restrict_particles(w,h)
            area_hist.append(filter.cur_a)
            pred_hist.append(np.array(gt_box).reshape(1,-1))
            #pca
            # test sample_iou
            num_true=500
            num_false=1000
            boxes_train = []
            #boxes_train_neg=[]
            iou_train = []
            try:
                #Q=[[1,0],[0,1]] #for pixel wise
                Q=0.05#box_w,box_h
                sample_box_true, sample_iou_true = filter.sample_iou(gt_box, Q, 0.01, 0.01, num_true, 0.8,
                                                                     1.0)
            except OverflowError as e:
                print "too many loops in sample."
            # print sample_box_true[:10]
            # print sample_box_true.shape[0]
            # print sample_iou_true[:10]
            print "average iou: ", np.mean(sample_iou_true)
            boxes_train.append(sample_box_true)
            iou_train.append(sample_iou_true)
            try:
                #Q=[[36,0],[0,36]]#for pixel wise
                Q=0.2#0.15
                sample_box_false, sample_iou_false = filter.sample_iou(gt_box, Q, 0.2, 0.01, num_false/2, 0,
                                                                       thre_max_neg)
            except OverflowError as e:
                print "too many loops in sample."
            # print sample_box_false[:10]
            # print sample_box_false.shape[0]
            # print sample_iou_false[:10]
            print "average iou: ", np.mean(sample_iou_false)
            boxes_train.append(sample_box_false)
            iou_train.append(sample_iou_false)
            try:
                #Q=[[36,0],[0,36]]#for pixel wise
                Q=0.2
                sample_box_false, sample_iou_false = filter.sample_iou(gt_box, Q, 0.01, 0.2, num_false/2, 0,
                                                                       thre_max_neg)
            except OverflowError as e:
                print "too many loops in sample."
            # print sample_box_false[:10]
            # print sample_box_false.shape[0]
            # print sample_iou_false[:10]
            print "average iou: ", np.mean(sample_iou_false)
            boxes_train.append(sample_box_false)
            iou_train.append(sample_iou_false)

            boxes_train = np.vstack(boxes_train)

            iou_train = np.vstack(iou_train)
            y_train_true=np.ones((num_true,))
            y_train_false=np.zeros((num_false,))

            y_train=np.hstack([y_train_true,y_train_false])

            #permutation
            ind_perm=np.random.permutation(range(num_false+num_true))
            boxes_train=boxes_train[ind_perm,:]
            iou_train=iou_train[ind_perm]
            y_train=y_train[ind_perm]
            ind_pos=np.where(y_train==1)[0]
            ind_neg=np.where(y_train==0)[0]


            vggnet.reshape(w=w,h=h,nbox=boxes_train.shape[0])
            features=vggnet.get_features_first_raw(frame_data,boxes_raw=boxes_train,id=id)

            for k,v in features.iteritems():
                print k,v.shape
                if k=='f3':
                    pca3=utils.skl_pca(v)
                    v_pca3=pca3.transform(v)
                    pca3_pos=np.zeros((num_true,pca3.n_components_),dtype=np.float32)
                    pca3_neg=np.zeros((num_false,pca3.n_components_),dtype=np.float32)
                    pca3_pos[...]=v_pca3[ind_pos,:]
                    pca3_neg[...]=v_pca3[ind_neg,:]
                    #utils.vis_as_image(v_pca3)
                    #plt.imshow(v_pca3)
                    #plt.title("PCA features")
                    #plt.show()
                    #plt.close()
                    #logistic regression
                    y_weight=sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=y_train)
                    #print y_weight
                    class_weight={0:y_weight[0],1:y_weight[1]}
                    clf3=linear_model.LogisticRegression(fit_intercept=True,solver='liblinear')
                    clf3.fit(v_pca3,y_train)
                    score3=clf3.score(v_pca3,y_train)
                    print 'score3: ',score3
                    #prob=clf3.predict_proba(v_pca3)
                    print clf3.classes_
                    #print prob

            vis_feature=False
            if vis_feature:
                utils.vis_features(features,id)

            start_time = time.time()
        else:
            if fail_times>=5:
                #reinitialize
                reinit+=1
                area=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
                ratio=(gt_box[2]-gt_box[0])/(gt_box[3]-gt_box[1])
                filter=PFfilter.PFfilter(utils.bbox_to_states(gt_box,area,ratio),area,ratio,w,h,nparticles)
                #filter.reset(utils.bbox_to_states(gt_box, area, ratio), area, ratio)
                filter.create_particles()
                filter.restrict_particles(w, h)
                area_hist.append(filter.cur_a)
                pred_box=gt_box
                boxes_train = []
                pred_hist.append(np.array(gt_box).reshape(1, -1))
                #pred_hist.append(pred_box)
                conf_hist.append(-0.1)

                # boxes_train_neg=[]
                iou_train = []
                try:
                    # Q=[[1,0],[0,1]] #for pixel wise
                    Q = 0.05  # box_w,box_h
                    sample_box_true, sample_iou_true = filter.sample_iou(gt_box, Q, 0.01, 0.01, num_true, 0.8,
                                                                         1.0)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_true[:10]
                # print sample_box_true.shape[0]
                # print sample_iou_true[:10]
                print "average iou: ", np.mean(sample_iou_true)
                boxes_train.append(sample_box_true)
                iou_train.append(sample_iou_true)
                try:
                    # Q=[[36,0],[0,36]]#for pixel wise
                    Q = 0.2#0.15
                    sample_box_false, sample_iou_false = filter.sample_iou(gt_box, Q, 0.2, 0.01, num_false / 2, 0,
                                                                           thre_max_neg)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_false[:10]
                # print sample_box_false.shape[0]
                # print sample_iou_false[:10]
                print "average iou: ", np.mean(sample_iou_false)
                boxes_train.append(sample_box_false)
                iou_train.append(sample_iou_false)
                try:
                    # Q=[[36,0],[0,36]]#for pixel wise
                    Q = 0.2
                    sample_box_false, sample_iou_false = filter.sample_iou(gt_box, Q, 0.01, 0.2, num_false / 2, 0,
                                                                           thre_max_neg)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_false[:10]
                # print sample_box_false.shape[0]
                # print sample_iou_false[:10]
                print "average iou: ", np.mean(sample_iou_false)
                boxes_train.append(sample_box_false)
                iou_train.append(sample_iou_false)

                boxes_train=np.vstack(boxes_train)
                iou_train=np.vstack(iou_train)
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

                for k, v in features.iteritems():
                    print k, v.shape
                    if k == 'f3':

                        v_pca3 = pca3.transform(v)

                        pca3_pos[...] = v_pca3[ind_pos, :]
                        pca3_neg[...] = v_pca3[ind_neg, :]
                        clf3.fit(v_pca3, y_train)
                        score3 = clf3.score(v_pca3, y_train)
                        print 'score3: ', score3
                        # prob=clf3.predict_proba(v_pca3)
                        print clf3.classes_
                fail_times=0
                continue

            filter.predict_particles(Q=0.2,cr=0.005,ca=0.001)#0.02,0.05,0.1
            filter.restrict_particles(w,h)
            area_hist.append(filter.cur_a)
            #compute conf
            conf=np.zeros(filter.weights.shape)
            #np.save('particles.npy',filter.particles)
            pred_boxes=utils.state_to_bbox(filter.particles,area,ratio)
            vggnet.reshape(w,h,filter.num_particles)
            features=vggnet.get_features_first_raw(frame_data,boxes_raw=pred_boxes,id=id)
            for k,v in features.iteritems():
                print k,v.shape
                if k=='f3':

                    v_pca3=pca3.transform(v)
                    #utils.vis_as_image(v_pca3)
                    #plt.imshow(v_pca3)
                    #plt.title("PCA features")
                    #plt.show()
                    #plt.close()
                    #logistic regression
                    conf=clf3.predict_proba(v_pca3)[:,1]





            conf_max = np.max(conf)
            conf_min = np.min(conf)
            print 'conf_max: ', conf_max
            print 'conf_min: ', conf_min
            filter.update_particles(conf)
            # pred_state = filter.estimate()

            filter.resample()  # always resample
            pred_state, s_particles, r_particles = filter.estimate(k=10)

            cov_particles = np.dot(filter.particles[:, :4].T, filter.particles[:, :4]) / filter.particles.shape[0]

            eigval,eigvec=np.linalg.eig(cov_particles)
            max_val=eigval[0]
            eig_hist.append(max_val)
            print 'Max eigvalue: %f'%max_val

            #print 'conf is: ',conf
            if conf_max>0.8:
                fail_times=0

            else:
                fail_times+=1
                #filter.update_particles(conf)
                #pred_state=filter.estimate()
                #filter.resample()
                #pred_state, s_particles, r_particles = filter.estimate(k=10)
                print "conf_max too low, not update particles "
            pred_box=utils.state_to_bbox(pred_state.reshape((-1,6)),area,ratio)
            print 'ground truth bbox is: ',gt_box
            print "pred_box is: ",pred_box
            show_sr=False
            if show_sr:
                plt.hist2d(s_particles,r_particles,bins=50,weights=filter.weights.squeeze())
                '''
                plt.scatter(s_particles,r_particles,c='r',marker='.',linewidths=1)
                plt.xlabel('Area')
                plt.ylabel('Aspect ratio')
                plt.title('Area and Ratio of particles')
                plt.axis('equal')
                '''
                plt.show()
            iou=utils.calc_iou(gt_box,pred_box)
            print 'iou is: ', iou
            pred_hist.append(pred_box)
            conf_hist.append(conf_max)
            iou_hist.append(iou)

            if conf_max>=0.9:#0.5
                #update pca3_pos and pca3_neg
                new_true=100#50
                new_false=200#100
                boxes_train = []

                iou_train = []
                Q = 0.02
                try:
                    sample_box_true, sample_iou_true = filter.sample_iou(pred_box, Q, 0.01, 0.01, new_true,
                                                                         0.85,
                                                                         1.0)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_true[:10]
                # print sample_box_true.shape[0]
                # print sample_iou_true[:10]
                print "average iou: ", np.mean(sample_iou_true)
                boxes_train.append(sample_box_true)
                iou_train.append(sample_iou_true)
                try:
                    Q = 0.2
                    sample_box_false, sample_iou_false = filter.sample_iou(pred_box, Q, 0.2, 0.01,
                                                                           new_false / 2, 0,
                                                                           thre_max_neg)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_false[:10]
                # print sample_box_false.shape[0]
                # print sample_iou_false[:10]
                print "average iou: ", np.mean(sample_iou_false)
                boxes_train.append(sample_box_false)
                iou_train.append(sample_iou_false)
                try:
                    Q = 0.2
                    sample_box_false, sample_iou_false = filter.sample_iou(pred_box, Q, 0.01, 0.2,
                                                                           new_false / 2, 0,
                                                                           thre_max_neg)
                except OverflowError as e:
                    print "too many loops in sample."
                # print sample_box_false[:10]
                # print sample_box_false.shape[0]
                # print sample_iou_false[:10]
                print "average iou: ", np.mean(sample_iou_false)
                boxes_train.append(sample_box_false)
                iou_train.append(sample_iou_false)

                boxes_train = np.vstack(boxes_train)


                iou_train = np.vstack(iou_train)
                y_train_true = np.ones((new_true,))
                y_train_false = np.zeros((new_false,))
                y_train = np.hstack([y_train_true, y_train_false])

                # permutation
                ind_perm = np.random.permutation(range(new_false + new_true))
                boxes_train = boxes_train[ind_perm, :]

                y_train = y_train[ind_perm]
                new_y=np.zeros(y_train.shape)
                new_y[...]=y_train
                ind_pos=np.where(y_train==1)[0]
                ind_neg=np.where(y_train==0)[0]

                vggnet.reshape(w=w, h=h, nbox=boxes_train.shape[0])
                features = vggnet.get_features_first_raw(frame_data, boxes_raw=boxes_train, id=id)
                for k, v in features.iteritems():
                    print k, v.shape
                    if k == 'f3':
                        v_pca3 = pca3.transform(v)

                        #random substitude
                        pca3_cur_pos=v_pca3[ind_pos,:]
                        pca3_cur_neg=v_pca3[ind_neg,:]
                        to_subst=random.sample(range(num_true),new_true)
                        pca3_pos[to_subst,:]=pca3_cur_pos
                        to_subst=random.sample(range(num_false),new_false)
                        pca3_neg[to_subst,:]=pca3_cur_neg



            if conf_max<1 and fail_times>=2:#0.9
            #if conf_max<0.95 and conf_max>0.5:
                #update classification model
                print 'updating model...'
                pca3_train=np.vstack([pca3_pos,pca3_neg])


                y_train_true = np.ones((num_true,))
                y_train_false = np.zeros((num_false,))
                y_train = np.hstack([y_train_true, y_train_false])

                # permutation
                ind_perm = np.random.permutation(range(num_false + num_true))
                pca3_train = pca3_train[ind_perm, :]

                y_train = y_train[ind_perm]

                #logistic regression

                clf3.fit(pca3_train,y_train)
                print 'score is: ',clf3.score(pca3_train,y_train)
                #fail_times=0


        # (B,G,R)
        frame_data_cv = frame_data * 255  # [0,1]-->[0,255]
        frame_data_cv = frame_data_cv[:, :, ::-1]  # RGB->BGR
        frame_data_cv = frame_data_cv.astype('uint8')
        #cv2.rectangle(frame_data_cv, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])),
         #                         (255, 0, 0), 2, 1)
        if id>0 and init_id==True:
            cv2.rectangle(frame_data_cv, (int(pred_box[0,0]), int(pred_box[0,1])), (int(pred_box[0,2]), int(pred_box[0,3])),
                          (0, 255, 0), 2, 1)
        if init_id==False:
           init_id=True
        show_particles = False
        if show_particles:
            for i in range(filter.num_particles):
                cx = filter.particles[i, 0]
                cy = filter.particles[i, 1]
                cv2.circle(frame_data_cv, (int(cx), int(cy)), 1, (0, 0, 255), thickness=1)
        show_box = False
        if show_box:
            n=0
            for i in ind_pos:
                if n%5==0:
                    cv2.rectangle(frame_data_cv, (int(boxes_train[i, 0]), int(boxes_train[i, 1])),
                                  (int(boxes_train[i, 2]), int(boxes_train[i, 3])), (0, 0, 255), 2, 1)
                n+=1
            n=0

        show_particles_init = False
        if show_particles_init:
            for i in range(filter.num_particles):
                cx = filter.particles[i, 0]
                cy = filter.particles[i, 1]
                cv2.circle(frame_data_cv, (int(cx), int(cy)), 1, (0, 255, 0), thickness=1)
        show_frame=False
        #cv2.circle(frame_data_cv, (int(filter.cur_c[0]), int(filter.cur_c[1])), 2, (0, 0, 255), thickness=1)

        if show_frame:
            cv2.imshow(sequence,frame_data_cv)

            c=cv2.waitKey(1)
            if c!=-1:
                if chr(c)=='p':
                    c=cv2.waitKey()
                #print 'You press: ',chr(c)
                #if chr(c)=='c':
                if chr(c)=='c':
                    cv2.destroyWindow(sequence)
                    #conf_hist=np.array(conf_hist)
                    #iou_hist=np.array(iou_hist)
                    #np.save('conf_hist.npy',conf_hist)
                    #np.save('iou_hist.npy',iou_hist)
                    break
        else:
            video_writer.write(frame_data_cv)
    end_time=time.time()
    video_writer.release()
    iou_hist=np.array(iou_hist)
    pred_hist=np.array(pred_hist).squeeze()
    print "iou_hist: ",iou_hist.shape
    print "pred_hist: ",pred_hist.shape
    print "get_boxes: ",gt_boxes.shape
    precisions, auc_pre=utils.calc_prec(gt_boxes,pred_hist)
    print "precision is: %f"%(precisions[19])
    suc, auc_iou=utils.calc_success(iou_hist)
    print "Average IOU is: %f"%(np.mean(iou_hist))
    print "Auc of precision is: %f"%(auc_pre)
    print "Auc of success is: %f"%auc_iou
    print "Reinit times: %d"%reinit
    print "Average FPS: %f" % ((id+1) / (end_time - start_time))
    #vggnet.net.save("vgg16_conv3.caffemodel")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save", action='store_true')
    #parser.add_argument("--prototxt", default='res152_align_res2c.prototxt', type=str)#res3b
    #parser.add_argument("--caffemodel", default='resnet/ResNet-152-model.caffemodel', type=str)
    #parser.add_argument("--prototxt", default='inceptionv2_align.prototxt', type=str)
    #parser.add_argument("--caffemodel", default='Inception21k.caffemodel', type=str)
    parser.add_argument("--prototxt", default='vgg16_align_conv3.prototxt', type=str)#vgg16_align_conv4.prototxt
    parser.add_argument("--caffemodel", default='VGG_ILSVRC_16_layers.caffemodel', type=str)
    #parser.add_argument("--prototxt", default='densenet/densenet121_align.prototxt', type=str)
    #parser.add_argument("--caffemodel", default='densenet/DenseNet_121.caffemodel', type=str)
    #parser.add_argument("--prototxt",default='darknet53_align.prototxt',type=str)
    #parser.add_argument("--caffemodel",default='yolov3.caffemodel',type=str)
    parser.add_argument("--sequence", default="Dog1", type=str)
    parser.add_argument("--classifier", default='svm', type=str)
    parser.add_argument("--particles", default=100, type=int)#256
    args = parser.parse_args()
    main(args)
