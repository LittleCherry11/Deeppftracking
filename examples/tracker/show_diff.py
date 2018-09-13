import numpy as np
import collections
import matplotlib.pyplot as plt
#data:(channel,width,height)
def vis_square(data,title):
    data_box=data
    #normalize data for display
    data_box=(data_box-data_box.min())/(data_box.max()-data_box.min())

    n=int(np.ceil(np.sqrt(data_box.shape[0])))
    padding=(((0,n**2-data_box.shape[0]),(0,1),(0,1))+((0,0),)*(data_box.ndim-3))
    data_box=np.pad(data_box,padding,mode='constant',constant_values=1)

    data_box=data_box.reshape((n,n)+data_box.shape[1:]).transpose((0,2,1,3)+tuple(range(4,data_box.ndim+1)))
    data_box=data_box.reshape((n*data_box.shape[1],n*data_box.shape[3])+data_box.shape[4:])
    plt.imshow(data_box)
    plt.title(title+"  diff")
    plt.axis('off')
    plt.show()

id=12
features_raw=np.load("features/basketball_features_100.npy")
features_dict = collections.OrderedDict()
for f in features_raw:
    features_dict.update(f)
# {'feature4': ,'feature5': ,'feature3': ,'im_info': }
for k, v in features_dict.items():
    if k == 'im_info': continue
    diff=v[1]-v[2]
    vis_square(diff, k)  # v:(nboxes=3, channel=256/512, 7,7)