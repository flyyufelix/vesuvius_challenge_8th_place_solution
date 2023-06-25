import prepare_data as yoyobar_prepare
import json,os
with open("./SETTINGS.json","r") as f:
    cfg=json.load(f)

if not os.path.exists(cfg["CLEAN_DATA_DIR"]):
    os.mkdir(cfg["CLEAN_DATA_DIR"])
yoyobar_prepare.main()
