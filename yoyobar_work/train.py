from train_sub import train_
from Modules import *

def main():
    #CFG.comp_dataset_path=PATH["TRAIN_DATA_CLEAN_PATH"]+"alex/"
    #CFG.model_dir=PATH["CHECKPOINT_DIR"]+"alex/"
    CFG.comp_dataset_path=PATH["TRAIN_DATA_CLEAN_PATH"]
    CFG.model_dir=PATH["CHECKPOINT_DIR"]
    
    if not os.path.exists(CFG.model_dir):
        os.mkdir(CFG.model_dir)
        
    final_model_path=PATH["MODEL_DIR"]+"se_resnet101.pth"

    CFG.valid_id='2a'


    CFG.label_size=96
    CFG.model_input_size=CFG.label_size+CFG.ex_size
    CFG.train_load_size=CFG.model_input_size
    CFG.stride = CFG.label_size // 2
    CFG.train_batch_size=128
    CFG.valid_batch_size=CFG.train_batch_size*2
    CFG.scheduler=True
    CFG.lr=2e-4
    CFG.chan_start=15 #16
    CFG.in_chans = 16 #12
    CFG.load_chans=30 #26
    CFG.total_per_epoch=40000
    CFG.epochs=50+20
    model,name=train_(CFG,break_fc=lambda x:x["ep"]>50,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}_init/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")
    
    ###############################################################
    
    CFG.label_size=192
    CFG.model_input_size=CFG.label_size+CFG.ex_size
    CFG.train_load_size=CFG.model_input_size
    CFG.stride = CFG.label_size // 2
    CFG.train_batch_size=32
    CFG.valid_batch_size=CFG.train_batch_size*2
    CFG.scheduler=False
    CFG.lr=2e-5
    CFG.chan_start=15 #16
    CFG.in_chans = 20 #12
    CFG.load_chans=30 #26
    CFG.total_per_epoch=20000
    CFG.epochs=100
    model,name=train_(CFG,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}_low_lr/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")

    ################################################################################
    CFG.label_size=256
    CFG.model_input_size=CFG.label_size+CFG.ex_size
    CFG.train_load_size=CFG.model_input_size
    CFG.stride = CFG.label_size // 2
    CFG.train_batch_size=24
    CFG.valid_batch_size=CFG.train_batch_size*2
    CFG.scheduler=False
    CFG.lr=2e-5
    CFG.chan_start=15
    CFG.in_chans = 20
    CFG.load_chans=30
    CFG.total_per_epoch=10000
    CFG.epochs=20
    model,name=train_(CFG,save_fc=lambda x:x%5==0)
    file_name=f"/final_v{CFG.valid_id}_lbs{CFG.label_size}/"
    if not os.path.exists(CFG.model_dir+file_name):
        os.mkdir(CFG.model_dir+file_name)
    tc.save(model.state_dict(), CFG.model_dir+file_name+name)
    os.system(f"rm {CFG.model_dir}*.pth")
    tc.save(model.state_dict(), CFG.model_dir+f"/now.pth")
    tc.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    main()
