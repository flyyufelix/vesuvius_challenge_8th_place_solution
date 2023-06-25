from .Modules import *

def load_raw_image(fragment_id):
    start = CFG.chan_start
    end = start + CFG.load_chans
    idxs = range(start, end)
    images=[]
    for i in tqdm(idxs):
        images.append(cv2.imread(CFG.comp_dataset_path + f"{fragment_id}/surface_volume/{i:02}{CFG.image_type}", 0))
    return images


def read_image_mask(cp_rate,raw_image,raw_mask):
    images = []
    for image in raw_image:
        if cp_rate!=1:
            image=cv2.resize(image,None,fx=cp_rate,fy=cp_rate,interpolation=cv2.INTER_NEAREST)

        pad0 = (CFG.model_input_size - image.shape[0] % CFG.model_input_size)
        pad1 = (CFG.model_input_size - image.shape[1] % CFG.model_input_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    ##########################################
    mask=raw_mask
    if cp_rate!=1:
        mask=cv2.resize(mask,None,fx=cp_rate,fy=cp_rate,interpolation=cv2.INTER_NEAREST)
        mask[mask!=255]=0
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask /= 255.0
    
    return images, mask

def fc1(image,mask,fragment_id,cp_rate,train_images,train_masks,valid_images,valid_masks,valid_xyxys):
    ex=(CFG.train_load_size-CFG.label_size)//2
    image = np.pad(image, ((ex, ex), (ex, ex),(0,0)), constant_values=0)#whold image位移

    x1_list = np.arange(0, image.shape[1]-CFG.train_load_size+1, CFG.stride)
    y1_list = np.arange(0, image.shape[0]-CFG.train_load_size+1, CFG.stride)
    for y1 in y1_list:
        for x1 in x1_list:
            x2 = x1 + CFG.label_size#label_size
            y2 = y1 + CFG.label_size

            x=image[y1:y2+2*ex,x1:x2+2*ex]#CFG.train_load_size
            label=mask[y1:y2, x1:x2, None]#label_size
            if np.all(x==0):
                continue
            assert label.shape[0]==CFG.label_size and label.shape[1]==CFG.label_size
            if fragment_id == CFG.valid_id:
                if cp_rate==CFG.val_cp_rate:
                    valid_images.append(x)
                    valid_masks.append(label)
                    valid_xyxys.append([x1, y1, x2, y2])
            else:
                train_images.append(x)
                train_masks.append(label)

def get_dataset(only_val=False):
    train_images = [list() for _ in range(len(CFG.cp_rate))]
    train_masks = [list() for _ in range(len(CFG.cp_rate))]

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in CFG.train_fragment_id:
        if only_val and fragment_id!=CFG.valid_id:
            continue
        raw_image=load_raw_image(fragment_id=fragment_id)
        raw_mask=cv2.imread(CFG.comp_dataset_path + f"{fragment_id}/inklabels.png", 0)
        for i,cp_rate in enumerate(CFG.cp_rate):
            if fragment_id == CFG.valid_id and cp_rate!=CFG.val_cp_rate:
                print(f"pass load val in cp_rate:{cp_rate}")
                continue
            image, mask = read_image_mask(cp_rate,raw_image,raw_mask)
            fc1(image,mask,fragment_id,cp_rate,train_images[i],train_masks[i],valid_images,valid_masks,valid_xyxys)
            print(f"fragment_id:{fragment_id}, cp_rate:{cp_rate}, train_cp_now:{len(train_images[i])}, total_val:{len(valid_images)}")
    return train_images, train_masks, [valid_images], [valid_masks], [valid_xyxys]

class CustomDataset(Dataset):
    def __init__(self, images_set,xys_set=None, labels_set=None,cfg=CFG,mode="train",total_per_epoch=40000):
        self.images_set = images_set
        self.labels_set = labels_set
        self.xys_set=xys_set
        self.cfg = copy.copy(cfg)
        self.cp_sample_rate=self.cfg.cp_sample_rate if mode=="train" else [1]
        self.cp_sample_rate=tc.tensor(self.cp_sample_rate,dtype=tc.float32)
        self.total_per_epoch=total_per_epoch
        
        if mode=="train":
            self.transform =CFG.train_aug
        else:
            self.transform =CFG.valid_aug
            assert len(self.images_set)==1
        self.mode=mode
        self.rotate=CFG.rotate
        
    
    def __len__(self)->int:
        if self.mode=="train":
            return self.total_per_epoch
        else:
            return len(self.images_set[0])

    def getitem(self,idx):
        dataset_index=tc.multinomial(self.cp_sample_rate,1)
        if self.mode=="train":
            idx=np.random.randint(len(self.images_set[dataset_index]))
        image = self.images_set[dataset_index][idx]
        label = self.labels_set[dataset_index][idx]

        ex=self.cfg.ex_size//2
        if ex!=0:
            label = np.pad(label, ((ex, ex), (ex, ex),(0,0)), constant_values=0)
        return image,label*255

    def __getitem__(self, idx):
        image,label=self.getitem(idx)
        
        if self.mode=="train":
            H,W,C=image.shape
            new_C=C+C*(np.random.rand()*2-1)*self.cfg.z_resize_rate
            new_C=int(new_C)
            image=cv2.resize(image.reshape(H*W,C),(new_C,H*W))
            image=image.reshape(H,W,new_C)
            
            #3d rotate
            #image.shape=(h,w,c)
            image=image.transpose(2,1,0)#(c,w,h)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,h,w)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,w,h)
            image=image.transpose(2,1,0)#(h,w,c)
        
        assert image.shape[-1]>=self.cfg.in_chans
        if image.shape[-1]-self.cfg.in_chans:
            if self.mode=="train":
                n=np.random.randint(image.shape[-1]-self.cfg.in_chans)
            else:
                n=int((image.shape[-1]-self.cfg.in_chans)/2)
            image=image[...,n:self.cfg.in_chans+n]#.astype(np.float32)
                
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']#shape=(C,H,F)
            label = data['mask']
            if self.mode=="train":
                #if np.random.rand()<0.3:
                #    image=torch.flip(image,dims=(0,))
                #    label*=0
                k=np.random.randint(4)
                image=torch.rot90(image,k=k,dims=(1,2))
                label=torch.rot90(label,k=k,dims=(1,2))

        ex=self.cfg.ex_size//2
        if ex!=0:
            label=label[:,ex:-ex,ex:-ex]
        image=image.to(tc.float32)/255
        image[image>0.78]=0.78
        label=(label.to(tc.float32)/255)#.mean(dim=(1,2))
        if self.mode=="train":
            if self.cfg.exponent_arg:
                image=min_max_normalization(image)
                image=image**(0.5+np.random.rand()*1.5)
                image=normalization(image)
            return image, label
        else: 
            return image, label,self.xys_set[0][idx]
