import numpy as np
import collections
import matplotlib.pyplot as plt


#data:(nboxes,channel,width,height)
def vis_square(data,title):
    data_box=data[0] #box0:(channel,height,width)
    #normalize data for display
    data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

    n=int(np.ceil(np.sqrt(data_box.shape[0])))
    padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
    data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

    data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
    data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
    plt.imshow(data_box)
    plt.title(title+"gt_box")
    plt.axis('off')
    plt.show()
    data_box=data[1] #box0:(channel,height,width)
    #normalize data for display
    data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

    n=int(np.ceil(np.sqrt(data_box.shape[0])))
    padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
    data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

    data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
    data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
    plt.imshow(data_box)
    plt.title(title+"gt_box-50")
    plt.axis('off')
    plt.show()

    data_box=data[2] #box0:(channel,height,width)
    #normalize data for display
    data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

    n=int(np.ceil(np.sqrt(data_box.shape[0])))
    padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
    data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

    data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
    data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
    plt.imshow(data_box)
    plt.title(title+"gt_box+50")
    plt.axis('off')
    plt.show()

features_raw=np.load("basketball_features.npy")
features_dict=collections.OrderedDict()
for f in features_raw:
    features_dict.update(f)
#{'feature4': ,'feature5': ,'feature3': ,'im_info': }
for k,v in features_dict.items():
    if k=='im_info': continue
    vis_square(v,k) #v:(nboxes=3, channel=256/512, 7,7)