from Modules import *
from Model import Model
from dataset import *

def train_(cfg=CFG,break_fc=lambda x:False,save_fc=lambda x:x%3==0):
    tc.backends.cudnn.enabled = True
    tc.backends.cudnn.benchmark = True

    print(cfg.label_size)
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    def criterion(y_pred:tc.Tensor, y_true):
        return (BCELoss(y_pred, y_true)+dice_coef_torch(y_pred, y_true))/2

    model = Model(cfg,loss_fc=criterion if not cfg.mean_output else nn.MSELoss())
    model.load_checkpoint(cfg.model_dir+f"/now.pth")
    model.check_bug()
    model = model.cuda()
    if cfg.scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(model.optimizer, max_lr=cfg.lr,
                                                        steps_per_epoch=20, epochs=3,
                                                        pct_start=0.1)


    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_dataset()
    valid_xyxys = np.stack(valid_xyxys)



    train_dataset = CustomDataset(train_images,None,train_masks,total_per_epoch= cfg.total_per_epoch)
    valid_dataset = CustomDataset(valid_images,valid_xyxys,valid_masks,mode="valid",total_per_epoch= cfg.total_per_epoch//4)

    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train_batch_size,
                            shuffle=True,persistent_workers=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                            )
    valid_loader = DataLoader(valid_dataset,
                            batch_size=cfg.valid_batch_size,
                            shuffle=True,persistent_workers=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    fragment_id = cfg.valid_id

    valid_mask_gt = cv2.imread(cfg.comp_dataset_path + f"{fragment_id}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (cfg.model_input_size - valid_mask_gt.shape[0] % cfg.model_input_size)
    pad1 = (cfg.model_input_size - valid_mask_gt.shape[1] % cfg.model_input_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    valid_mask_gt=tc.from_numpy(valid_mask_gt)

    for epoch in range(1, cfg.epochs+1):
        model.train()
        cur_lr = f"LR : {model.optimizer.param_groups[0]['lr']:.2E}"
        time_=tqdm(total=len(train_loader)*train_loader.batch_size)
        mloss_train, mloss_val, val_metric = 0.0, 0.0, 0.0
        
        for i, (fragments, masks) in enumerate(train_loader):
            fragments, masks = fragments.cuda(), masks.cuda()
            fragments=add_noise(fragments,np.random.random()*cfg.noise_rate+0.01)
            output=model.AMP_forward(fragments)
            loss=model.get_loss(output,masks)
            model.AMP_backward(loss,max_grad_norm=cfg.max_grad_norm)
            mloss_train += loss.detach().item()

            time_.set_description(f"Epoch:{epoch}/{cfg.epochs} {cur_lr} loss:{mloss_train / (i + 1):.4f}")
            time_.update(fragments.shape[0])
        time_.close()
        mloss_train/=(i + 1)
            
        if cfg.scheduler:
            scheduler.step()
        ######################################################
        time_=tqdm(total=len(valid_loader)*valid_loader.batch_size)
        #model.eval()
        
        mask_pred = tc.zeros(int(valid_mask_gt.shape[0]*cfg.val_cp_rate),int(valid_mask_gt.shape[1]*cfg.val_cp_rate))
        mask_count = tc.zeros(int(valid_mask_gt.shape[0]*cfg.val_cp_rate),int(valid_mask_gt.shape[1]*cfg.val_cp_rate))
        
        for i, (fragments, masks, xys) in enumerate(valid_loader):
            fragments, masks = fragments.cuda(), masks.cuda()
            with torch.no_grad():
                pred_masks = model.AMP_forward(fragments)
                mloss_val += model.get_loss(pred_masks,masks).item()
                pred_masks=model.predict(model_output=pred_masks)
            for k, (x1, y1, x2, y2) in enumerate(xys):
                if mask_pred[y1:y2, x1:x2].shape!=(cfg.label_size,cfg.label_size):
                    #print("pass")
                    continue
            
                mask_pred[y1:y2, x1:x2] += pred_masks[k].squeeze(0).cpu()
                mask_count[y1:y2, x1:x2] += 1
            
            time_.set_description(f"Val Loss: {mloss_val / (i+1):.4f}")
            time_.update(fragments.shape[0])
        time_.close()
        mloss_val/=(i + 1)

        mask_pred/=(mask_count+1e-7)
        if cfg.val_cp_rate!=1:
            mask_pred=cv2.resize(mask_pred.numpy(),(valid_mask_gt.shape[1],valid_mask_gt.shape[0]),interpolation=cv2.INTER_NEAREST)
            mask_pred=tc.from_numpy(mask_pred)
            
        bast_fbeta=0
        for threshold in np.arange(0.3, 0.85, 0.05):
            fbeta=fbeta_score(mask_pred,valid_mask_gt,threshold)
            if bast_fbeta<fbeta:
                bast_fbeta=fbeta
            print(f"Threshold : {threshold:.2f}\tFBeta : {fbeta:.6f}")

        if save_fc(epoch):
            torch.save(model.state_dict(), cfg.model_dir+f"/{cfg.backbone}_epoch_{epoch}_tl{mloss_train:.3f}_vl{mloss_val:.3f}_vp{bast_fbeta:.3f}.pth")
        torch.save(model.state_dict(), cfg.model_dir+f"/now.pth")

        torch.cuda.empty_cache()

        if break_fc({"tl":mloss_train,"vl":mloss_val,"vp":bast_fbeta,"ep":epoch}):
            return model,f"/{cfg.backbone}_epoch_{epoch}_tl{mloss_train:.3f}_vl{mloss_val:.3f}_vp{bast_fbeta:.3f}.pth"
    return model,f"/{cfg.backbone}_epoch_{epoch}_tl{mloss_train:.3f}_vl{mloss_val:.3f}_vp{bast_fbeta:.3f}.pth"

if __name__=="__main__":
    train_()
