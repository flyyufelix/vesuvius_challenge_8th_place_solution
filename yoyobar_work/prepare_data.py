from Modules import *

def main():
    dataset_path=PATH["RAW_DATA_DIR"]
    #target=PATH["CLEAN_DATA_DIR"]+"alex/"
    target=PATH["CLEAN_DATA_DIR"]
    a=6000
    
    if not os.path.exists(target):
        os.makedirs(target)


    for i in ["2a","2b","1","3"]:
        if not os.path.exists(target + f"train/{i}"):
            os.makedirs(target + f"train/{i}")
        if not os.path.exists(target + f"train/{i}/surface_volume"):
            os.makedirs(target + f"train/{i}/surface_volume")



    for i in ["1","3"]:
        paths=glob(dataset_path + f"train/{i}/surface_volume/*.tif")
        paths.sort()
        label = cv2.imread(dataset_path + f"train/{i}/inklabels.png", 0)
        cv2.imwrite(target + f"train/{i}/inklabels.png", label, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        for path in tqdm(paths):
            index=path.split('/')[-1].split('.')[0]
            image = cv2.imread(path, 0)
            cv2.imwrite(target + f"train/{i}/surface_volume/{index}.png", image[:a], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    

    paths=glob(dataset_path + f"train/2/surface_volume/*.tif")
    paths.sort()
    label = cv2.imread(dataset_path + f"train/2/inklabels.png", 0)

    print("2a:",tc.from_numpy(label[:a]).numel())
    print("2b:",tc.from_numpy(label[a:]).numel())

    cv2.imwrite(target + f"train/2a/inklabels.png", label[:a], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(target + f"train/2b/inklabels.png", label[a:], [cv2.IMWRITE_PNG_COMPRESSION, 3])

    for path in tqdm(paths):
        index=path.split('/')[-1].split('.')[0]
        image = cv2.imread(path, 0)
        cv2.imwrite(target + f"train/2a/surface_volume/{index}.png", image[:a], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(target + f"train/2b/surface_volume/{index}.png", image[a:], [cv2.IMWRITE_PNG_COMPRESSION, 3])

    

if __name__ == "__main__":
    main()
