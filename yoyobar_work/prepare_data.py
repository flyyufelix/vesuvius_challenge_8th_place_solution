from Modules import *

def main(json_path="./SETTINGS.json"):
    with open(json_path,"r") as f:
        cfg=json.load(f)
    dataset_path=cfg["RAW_DATA_DIR"]
    target=cfg["CLEAN_DATA_DIR"]+"/alex/"
    a=6000



    for i in ["2a","2b"]:
        if not os.path.exists(target + f"train/{i}"):
            os.mkdir(target + f"train/{i}")
        if not os.path.exists(target + f"train/{i}/surface_volume"):
            os.mkdir(target + f"train/{i}/surface_volume")

    paths=glob(dataset_path + f"train/2/surface_volume/*.tif")
    paths.sort()
    label = cv2.imread(dataset_path + f"train/2/inklabels.png", 0)

    print(tc.from_numpy(label[:a]).numel())
    print(tc.from_numpy(label[a:]).numel())

    print(label.shape)

    cv2.imwrite(target + f"train/2a/inklabels.png", label[:a], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(target + f"train/2b/inklabels.png", label[a:], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    for path in tqdm(paths):
        index=path.split('.')[0].split("/")[-1]
        image = cv2.imread(path, 0)
        cv2.imwrite(target + f"train/2a/surface_volume/{index}.png", image[:a], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(target + f"train/2b/surface_volume/{index}.png", image[a:], [cv2.IMWRITE_PNG_COMPRESSION, 9])
