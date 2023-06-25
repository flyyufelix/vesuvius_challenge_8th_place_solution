from .Modules import *
from .dataset import *
from .Model import *

def get(model_name="/test.pth",fc=lambda x:x):
    model = Model(CFG)
    model.load_checkpoint(CFG.model_dir+model_name)
    model = model.cuda().eval()

    fragment_id = CFG.valid_id

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"{fragment_id}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.model_input_size - valid_mask_gt.shape[0] % CFG.model_input_size)
    pad1 = (CFG.model_input_size - valid_mask_gt.shape[1] % CFG.model_input_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    valid_mask_gt=tc.from_numpy(valid_mask_gt)



    _, _, valid_images, valid_masks, valid_xyxys = get_dataset(only_val=True)
    valid_xyxys = np.stack(valid_xyxys)

    valid_dataset = CustomDataset(valid_images,valid_xyxys,valid_masks,mode="valid")

    valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    # eval
    mask_pred = tc.zeros(int(valid_mask_gt.shape[0]*CFG.val_cp_rate),int(valid_mask_gt.shape[1]*CFG.val_cp_rate))
    mask_count = tc.zeros(int(valid_mask_gt.shape[0]*CFG.val_cp_rate),int(valid_mask_gt.shape[1]*CFG.val_cp_rate))
    time_=tqdm(total=len(valid_loader)*valid_loader.batch_size)
    for i, (fragments, masks, xys) in enumerate(valid_loader):
        fragments=fragments.cuda()
        fragments=fc(fragments)
        with torch.no_grad():
            pred_masks=TTA(fragments,model).cpu()

        for k, (x1, y1, x2, y2) in enumerate(xys):
            if mask_pred[y1:y2, x1:x2].shape!=(CFG.label_size,CFG.label_size):
                continue
            mask_pred[y1:y2, x1:x2] += pred_masks[k][0]#.reshape(mask_pred[y1:y2, x1:x2].shape)
            mask_count[y1:y2, x1:x2] += 1
        
        time_.update(fragments.shape[0])
    time_.close()

    mask_pred/=(mask_count+1e-7)
    if CFG.val_cp_rate!=1:
        mask_pred=cv2.resize(mask_pred.numpy(),(valid_mask_gt.shape[1],valid_mask_gt.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask_pred=tc.from_numpy(mask_pred)
    
    ################################################
    bast_fbeta=0
    for threshold in np.arange(0.1, 0.95, 0.05):
        fbeta=fbeta_score(mask_pred,valid_mask_gt,threshold)
        if bast_fbeta<fbeta:
            bast_fbeta=fbeta
        print(f"Threshold : {threshold:.2f}\tFBeta : {fbeta:.6f}")
    print(f"auc:{nn_class.count_auc(mask_pred.cuda(),valid_mask_gt.cuda())}, F0.5:{bast_fbeta}")
    mask_pred=(mask_pred*255).to(tc.uint8)
    tc.save(mask_pred,CFG.model_dir+f"{'.'.join(model_name.split('.')[:-1])}_val{CFG.valid_id}_xgb_input.pt")
    tc.save(valid_mask_gt,CFG.model_dir+f"val{CFG.valid_id}_xgb_mask.pt")
    return mask_pred,valid_mask_gt


if __name__=="__main__":
    CFG.backbone = 'SE-resnet3d-101'
    CFG.label_size=96
    CFG.ex_size = 0

    CFG.model_input_size=CFG.label_size+CFG.ex_size
    CFG.train_load_size=CFG.model_input_size
    CFG.stride = CFG.label_size // 2
    CFG.valid_batch_size = 8
    CFG.chan_start=20 #12  
    CFG.in_chans = 20 #12
    CFG.load_chans=20 #26
    CFG.TTA=False
    CFG.valid_id = "2a"
    CFG.train_fragment_id = [CFG.valid_id]
    CFG.mean_output = False
    CFG.val_cp_rate = 1
    CFG.cp_rate = [CFG.val_cp_rate]
    cache=0
    for model_name in ["test.pth"]:
        mask_pred,valid_mask_gt=get(model_name)

    ########################################################
    #exit()
    mask_pred=mask_pred.numpy()
    valid_mask_gt=valid_mask_gt.numpy()
    kde = np.histogram(mask_pred[valid_mask_gt!=0].reshape(-1), bins=200, density=True)
    kde_x, kde_y = kde[1], kde[0]
    plt.plot(kde_x[:-1], kde_y, label='ink')
    plt.show()
    plt.subplot(121)
    plt.imshow(mask_pred)
    plt.subplot(122)
    plt.imshow(valid_mask_gt)
    plt.show()
    exit()