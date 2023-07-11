from PIL import Image, ImageOps
import numpy as np
import os


def main():
    path = "./test/kkanji2"
    DATASET = []
    LABELS = []
    LABELS_INTT = []
    i = 0
    for dir_name in os.listdir(path):
        file_id = dir_name
        for filename in os.listdir(f"{path}/{dir_name}"):
            img_path = f"{path}/{dir_name}/{filename}"
            image = Image.open(img_path)
            image = ImageOps.grayscale(image)
            image = image.resize((28, 28))
            img_array = np.array(image)
            DATASET.append(img_array)
            LABELS.append(str(file_id))
            LABELS_INTT.append(i)
        i += 1

    DATASET = np.array(DATASET)
    np.save("./np_dataset.npy", DATASET)
    LABELS = np.array(LABELS)
    np.save("./labels.npy", LABELS)
    LABELS_INTT = np.array(LABELS_INTT)
    np.save("./labels_intt.npy", LABELS_INTT)


main()
