from .train_sub import train_
from .Modules import *

def main(json_path="./SETTINGS.json"):
    with open(json_path,"r") as f:
        cfg=json.load(f)
    CFG.comp_dataset_path=cfg["TRAIN_DATA_CLEAN_PATH"]+"alex/"
    CFG.model_dir=cfg["CHECKPOINT_DIR"]+"alex/"
    
    if not os.path.exists(CFG.model_dir):
        os.mkdir(CFG.model_dir)
        
    final_model_path=cfg["MODEL_DIR"]="se_resnet101.pth"

    CFG.valid_id='2a'


    CFG.label_size=96
    CFG.train_batch_size=128
    CFG.scheduler=True
    CFG.lr=2e-4
    CFG.chan_start=15 #16
    CFG.in_chans = 16 #12
    CFG.load_chans=30 #26
    CFG.total_per_epoch=40000
    CFG.epochs=70
    model,name=train_(CFG,break_fc=lambda x:x["ep"]>50,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}_init/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")
    
    ###############################################################
    
    CFG.label_size=96
    CFG.train_batch_size=128
    CFG.scheduler=False
    CFG.lr=2e-5
    CFG.chan_start=15 #16
    CFG.in_chans = 20 #12
    CFG.load_chans=30 #26
    CFG.total_per_epoch=40000
    CFG.epochs=150
    model,name=train_(CFG,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}_low_lr/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")

    ################################################################################
    CFG.label_size=256
    CFG.train_batch_size=24
    CFG.scheduler=False
    CFG.lr=2e-5
    CFG.chan_start=15
    CFG.in_chans = 20
    CFG.load_chans=30
    CFG.total_per_epoch=20000
    CFG.epochs=20
    model,name=train_(CFG,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")
    tc.save(model.state_dict(), final_model_path)
