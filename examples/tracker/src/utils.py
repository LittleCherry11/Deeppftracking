import numpy as np
import  re
import matplotlib.pyplot as plt
from scipy.stats import  multivariate_normal
import sklearn
from sklearn.decomposition import PCA,KernelPCA

eps=1e-10
def get_boxes_all(gt_path):
    try:
        gt_str=np.loadtxt(gt_path,dtype=str,delimiter='\n')
        gt_boxes=map(lambda x:map(float,re.split('[, \t]',x)),gt_str)
        gt_boxes=np.array(gt_boxes,dtype=float)
        gt_boxes[:,2:]=gt_boxes[:,:2]+gt_boxes[:,2:] #(x1,y1,w,h)->(x1,y1,x2,y2)
    except IOError:
        print 'Fail to open ',gt_path
    return gt_boxes

# bbox: (x1,y1,x2,y2)
def bbox_to_states_m(bbox, area, ratio):
    '''(x1,y1,x2,y2)->(cx,cy,s,r,dcx,dcy)'''
    bbox = bbox.reshape((-1, 4))
    # np.save('bbox.npy',bbox)
    box_w = bbox[:, 2] - bbox[:, 0]
    box_h = bbox[:, 3] - bbox[:, 1]
    cx = bbox[:, 0] + box_w / 2
    cy = bbox[:, 1] + box_h / 2
    s = box_w * box_h / area
    #print box_w / (box_h + eps)
    #print ratio
    r = np.log(box_w / (box_h + eps) + eps) / np.log(ratio)

    states = np.zeros((bbox.shape[0], 6))
    states[:, 0:4] = np.vstack((cx, cy, s, r)).transpose()

    return states
# bbox: (x1,y1,x2,y2)
def bbox_to_states(bbox, area, ratio):
    '''(x1,y1,x2,y2)->(cx,cy,s,r,dcx,dcy)'''
    bbox = bbox.reshape((-1, 4))
    # np.save('bbox.npy',bbox)
    box_w = bbox[:, 2] - bbox[:, 0]
    box_h = bbox[:, 3] - bbox[:, 1]
    cx = bbox[:, 0] + box_w / 2
    cy = bbox[:, 1] + box_h / 2
    s = box_w * box_h / area
    #print box_w / (box_h + eps)
    #print ratio
    r = box_w/(box_h+eps)/ratio

    states = np.zeros((bbox.shape[0], 6))
    states[:, 0:4] = np.vstack((cx, cy, s, r)).transpose()

    return states

def state_to_bbox_m(state, area, ratio):
    '''(cx,cy,s,r,dcx,dcy)->(x1,y1,x2,y2)'''
    state=state.reshape((-1,6))
    s = state[:, 2] * area
    # print s
    r = np.power(ratio, state[:, 3])
    # print r
    box_w = np.sqrt(r * s)
    # print box_w
    box_h = np.sqrt(s / (r + eps))
    # print box_h
    x1 = state[:, 0] - box_w / 2
    y1 = state[:, 1] - box_h / 2
    x2 = state[:, 0] + box_w / 2
    y2 = state[:, 1] + box_h / 2

    bbox = np.vstack((x1, y1, x2, y2)).transpose()
    # print bbox
    return bbox
def state_to_bbox(state, area, ratio):
    '''(cx,cy,s,r,dcx,dcy)->(x1,y1,x2,y2)'''
    state=state.reshape((-1,6))
    s = np.maximum(eps,state[:,2])* area
    # print s
    r = np.maximum(eps,state[:,3])*ratio
    # print r
    box_w = np.sqrt(r * s)
    # print box_w
    box_h = np.sqrt(s / (r + eps))
    # print box_h
    x1 = state[:, 0] - box_w / 2
    y1 = state[:, 1] - box_h / 2
    x2 = state[:, 0] + box_w / 2
    y2 = state[:, 1] + box_h / 2

    bbox = np.vstack((x1, y1, x2, y2)).transpose()
    # print bbox
    return bbox

def calc_iou(gt_box, pred_box):
    '''gt_box: (1,4)  pred_box: (N,4)'''
    gt_box = gt_box.reshape((1, 4))
    iou = np.zeros((pred_box.shape[0], 1))
    x1 = np.maximum(gt_box[:, 0], pred_box[:, 0])
    x2 = np.minimum(gt_box[:, 2], pred_box[:, 2])
    y1 = np.maximum(gt_box[:, 1], pred_box[:, 1])
    y2 = np.minimum(gt_box[:, 3], pred_box[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1]) + (pred_box[:, 2] - pred_box[:, 0]) * (
                pred_box[:, 3] - pred_box[:, 1])
    intersection = intersection.astype(np.float)
    iou = intersection / (union - intersection + eps)
    return iou

def calc_iou_all(gt_box, pred_box):
    '''gt_box: (1,4)  pred_box: (N,4)'''

    iou = np.zeros((pred_box.shape[0], 1))
    x1 = np.maximum(gt_box[:, 0], pred_box[:, 0])
    x2 = np.minimum(gt_box[:, 2], pred_box[:, 2])
    y1 = np.maximum(gt_box[:, 1], pred_box[:, 1])
    y2 = np.minimum(gt_box[:, 3], pred_box[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1]) + (pred_box[:, 2] - pred_box[:, 0]) * (
                pred_box[:, 3] - pred_box[:, 1])
    intersection = intersection.astype(np.float)
    iou = intersection / (union - intersection + eps)
    return iou
def restrict_box(boxes,w,h):
    '''box: (x1,y1,x2,y2)'''
    if len(boxes.shape)==1:
        boxes[0]=np.minimum(np.maximum(0,boxes[0]),w)
        boxes[2]=np.minimum(np.maximum(0,boxes[2]),w)
        boxes[1]=np.minimum(np.maximum(0,boxes[1]),h)
        boxes[3]=np.minimum(np.maximum(0,boxes[3]),h)
    else:
        boxes[:,0]=np.minimum(np.maximum(0,boxes[:,0]),w)
        boxes[:,2]=np.minimum(np.maximum(0,boxes[:,2]),w)
        boxes[:,1]=np.minimum(np.maximum(0,boxes[:,1]),h)
        boxes[:,3]=np.minimum(np.maximum(0,boxes[:,3]),h)
    return boxes

def vis_as_image(features):
    '''features: [nsamples,nfeature]'''
    N=features.shape[0]
    #fig=plt.figure(0,figsize=(10,10))
    '''
    plt.subplot(121)
    plt.imshow(features[:N/2])
    plt.subplot(122)
    plt.imshow(features[N/2:])
    '''
    plt.imshow(features)
    plt.title("PCA features")
    plt.show()
    plt.close()


def vis_features(features,id):
    '''features: {'f3': ,'f4': ,'f5': },value shape: (nbox,channel,height,width'''

    def callback(event):
        if event.key=='c':
            plt.close(fig)


    nbox=features['f3'].shape[0]
    for i in range(nbox):
        if i%50==0:
            data_box=features['f3'][i] #box0:(channel,height,width)
            data_box=data_box.reshape((256,7,7))
            #normalize data for display
            data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

            n=int(np.ceil(np.sqrt(data_box.shape[0])))
            padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
            data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

            data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
            data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])

            fig=plt.figure(1)
            fig.canvas.mpl_connect('key_press_event',callback)
            plt.imshow(data_box)
            plt.title("particles %d, conv3"%i)
            plt.axis('off')
            plt.show()
            #plt.pause(3)#wait for 3 seconds
            data_box=features['f4'][i] #box0:(channel,height,width)
            data_box=data_box.reshape((512,7,7))
            #normalize data for display
            data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

            n=int(np.ceil(np.sqrt(data_box.shape[0])))
            padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
            data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

            data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
            data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
            fig=plt.figure(1)
            fig.canvas.mpl_connect('key_press_event',callback)
            plt.imshow(data_box)
            plt.title("particles %d, conv4"%i)
            plt.axis('off')
            plt.show()
            data_box=features['f5'][i] #box0:(channel,height,width)
            data_box=data_box.reshape((512,7,7))
            #normalize data for display
            data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

            n=int(np.ceil(np.sqrt(data_box.shape[0])))
            padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
            data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

            data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
            data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
            fig=plt.figure(1)
            fig.canvas.mpl_connect('key_press_event',callback)
            plt.imshow(data_box)
            plt.title("particles %d, conv5"%i)
            plt.axis('off')
            plt.show()


def calc_pdf(gt_box, pred_boxes, s):
    w = gt_box[2] - gt_box[0]
    h = gt_box[3] - gt_box[1]
    d = np.array([[w * w, 0, 0, 0], [0, h * h, 0, 0], [0, 0, w * w, 0], [0, 0, 0, h * h]], dtype=np.float32)
    s = 0.01
    cov = d * s
    pdf = multivariate_normal.pdf(pred_boxes, mean=gt_box, cov=cov)
    return pdf
def save_box(gt_box,pred_boxes,pdf,name):
    '''(x1,y1,x2,y2,pdf,x1_g,y1_g,x2_g,y2_g)'''



    N=pred_boxes.shape[0]
    pdf=pdf.reshape((N,1))
    #scale pdf to (0,100)
    p_min=np.min(pdf)
    p_max=np.max(pdf)
    pdf_n=(pdf-p_min)/(p_max-p_min)*100
    g_rep=np.tile(gt_box,N).reshape((N,4))
    box_all=np.hstack([pred_boxes,pdf_n,g_rep])
    #print box_all[:5]
    np.save(name,box_all)

def calc_pca(features):
    '''features: [nsamples,feature_dim]'''
    mean_val=np.mean(features,axis=0)
    features=features-mean_val
    cov=np.cov(features,rowvar=False)
    eigvalue,eigvector=np.linalg.eig(cov)
    ind=np.argsort(-eigvalue)
    #print type(ind)
    eigvalue=eigvalue[ind[0]]
    eigvector=eigvector[:,ind[0]]

def skl_pca(features):
    '''do PCA using sklearn.PCA'''
    need_norm=1  #whether should normalize first
    if need_norm==1:
        #mean=0,unit covariance
        features=sklearn.preprocessing.scale(features,axis=0,with_mean=True,with_std=True)
    if need_norm==2:
        #normalize  to [0,1]
        min_perfeat=np.min(features,axis=0)
        max_perfeat=np.max(features,axis=0)

        features=(features-min_perfeat)/(max_perfeat-min_perfeat+eps)
    pca=PCA(n_components=0.95,svd_solver='full',whiten=False)#whiten=True,perform whiten,0.95
    pca.fit(features)
    print 'PCA components: %d'%pca.n_components_
    #print pca.explained_variance_
    #print pca.explained_variance_ratio_
    return pca
def skl_pca2(features):
    '''do PCA using sklearn.PCA'''
    need_norm=1  #whether should normalize first
    if need_norm==1:
        #mean=0,unit covariance
        scaler=sklearn.preprocessing.StandardScaler(with_mean=True,with_std=True)
        scaler.fit(features)
        features=scaler.transform(features)
        #features=sklearn.preprocessing.scale(features,axis=0,with_mean=True,with_std=True)
    if need_norm==2:
        #normalize  to [0,1]
        min_perfeat=np.min(features,axis=0)
        max_perfeat=np.max(features,axis=0)

        features=(features-min_perfeat)/(max_perfeat-min_perfeat+eps)
    pca=PCA(n_components=0.9,svd_solver='full',whiten=False)#whiten=True,perform whiten,0.9 , 'full'
    pca.fit(features)
    print 'PCA components: %d'%pca.n_components_
    #print pca.explained_variance_
    #print pca.explained_variance_ratio_
    return pca,scaler,pca.n_components_
def skl_pca_noscale(features):
    '''do PCA using sklearn.PCA'''

    pca=PCA(n_components=0.9,svd_solver='full',whiten=False)#whiten=True,perform whiten,0.9 , 'full',128
    pca.fit(features)
    print 'PCA components: %d'%pca.n_components_
    #print pca.explained_variance_
    #print pca.explained_variance_ratio_
    return pca,pca.n_components_
def skl_Kpca_noscale(features):
    '''do PCA using sklearn.PCA'''

    pca=KernelPCA(n_components=128,kernel='rbf')#whiten=True,perform whiten,0.9 , 'full',128
    pca.fit(features)
    #print 'PCA components: %d'%pca.n_components_
    #print pca.explained_variance_
    #print pca.explained_variance_ratio_
    return pca
def skl_modelselect(features,labels):
    lsvc=sklearn.svm.LinearSVC(C=0.01,penalty='l1',dual=True).fit(features,labels)#nsamples<nfeatures
    model=sklearn.feature_selection.SelectFromModel(lsvc,prefit=True)
    features_new=model.transform(features)
    print features_new.shape

def calc_prec(gt_box,pred_box):
    '''Calculate precision for a series of distances thresholds(percentage of frames where
    the distance to the gt_box is within the threshold. gt_box and pred_box are all Nx2
    '''
    max_threshold=50
    #gt_box=gt_box[1:,...]
    precisions=np.zeros((max_threshold,1),dtype=np.float32)
    N=np.minimum(gt_box.shape[0],pred_box.shape[0])
    c_box=np.zeros((N,2),dtype=np.float32)
    c_box[:N,0]=pred_box[:N,0]+(pred_box[:N,2]-pred_box[:N,0])/2.0
    c_box[:N,1]=pred_box[:N,1]+(pred_box[:N,3]-pred_box[:N,1])/2.0

    g_box = np.zeros((N, 2), dtype=np.float32)
    g_box[:N, 0] = gt_box[:N, 0] + (gt_box[:N, 2] - gt_box[:N, 0]) / 2.0
    g_box[:N, 1] = gt_box[:N, 1] + (gt_box[:N, 3] - gt_box[:N, 1]) / 2.0
    distances=np.zeros((N,),dtype=np.float32)
    distances[:]=np.sqrt(np.square(c_box[:N,0]-g_box[:N,0])+np.square(c_box[:N,1]-g_box[:N,1]))
    for i in range(max_threshold):
        precisions[i]=np.where(distances<=(i+1))[0].shape[0]/float(N)
    auc=np.sum(precisions)/float(max_threshold)
    return precisions,auc
def calc_success(iou):
    N=101
    n_iou=iou.shape[0]
    suc=np.zeros((N,),dtype=np.float32)
    x=np.linspace(0,1.0,N)
    for i in range(N):
        suc[i]=np.where(iou>=x[i])[0].shape[0]/float(n_iou)
    auc=np.sum(suc)/float(N)
    return suc,auc