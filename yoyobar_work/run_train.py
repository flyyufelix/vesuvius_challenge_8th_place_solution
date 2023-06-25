import train as yoyobar_train
import json,os

with open("./SETTINGS.json","r") as f:
    cfg=json.load(f)

if not os.path.exists(cfg["CHECKPOINT_DIR"]):
    os.mkdir(cfg["CHECKPOINT_DIR"])
if not os.path.exists(cfg["MODEL_DIR"]):
    os.mkdir(cfg["MODEL_DIR"])
yoyobar_train.main()
