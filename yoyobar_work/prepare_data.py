from Modules import *

def main():
    dataset_path=PATH["RAW_DATA_DIR"]
    #target=PATH["CLEAN_DATA_DIR"]+"alex/"
    target=PATH["CLEAN_DATA_DIR"]
    a=6000
    
    if not os.path.exists(target):
        os.makedirs(target)


    for i in ["2a","2b"]:
        if not os.path.exists(target + f"train/{i}"):
            os.makedirs(target + f"train/{i}")
        if not os.path.exists(target + f"train/{i}/surface_volume"):
            os.makedirs(target + f"train/{i}/surface_volume")

    paths=glob(dataset_path + f"train/2/surface_volume/*.tif")
    paths.sort()
    label = cv2.imread(dataset_path + f"train/2/inklabels.png", 0)

    print(tc.from_numpy(label[:a]).numel())
    print(tc.from_numpy(label[a:]).numel())

    print(label.shape)

    cv2.imwrite(target + f"train/2a/inklabels.png", label[:a], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(target + f"train/2b/inklabels.png", label[a:], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    for path in tqdm(paths):
        index=path.split('/')[-1].split('.')[0]
        image = cv2.imread(path, 0)
        cv2.imwrite(target + f"train/2a/surface_volume/{index}.png", image[:a], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(target + f"train/2b/surface_volume/{index}.png", image[a:], [cv2.IMWRITE_PNG_COMPRESSION, 9])

if __name__ == "__main__":
    main()
