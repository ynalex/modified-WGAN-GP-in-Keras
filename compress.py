import numpy as np
import cv2
import os
import random

def compress_image(image_list, mutiplier):
    image_list = np.array(image_list)

    h = int(image_list.shape[0])
    w = int(image_list.shape[1])
    new_h = int(h/mutiplier)
    new_w = int(w/mutiplier)

    print("Compressing images with size" + " ({},".format(h) + "{}".format(w) + ").")
    
    new_image = np.zeros([new_h, new_w, 3])

    for i in range(int(h/mutiplier)):
        for j in range(int(w/mutiplier)):
                for z in range(3):
                    new_image[int(i),int(j),z] = image_list[i*mutiplier,j*mutiplier,z]

    return new_image


if __name__ == "__main__":
    mutiplier = 4
    image_source = 'source'
    os.makedirs('image', exist_ok=True)
    for file in sorted(os.listdir(image_source)):
        image_path = os.path.join(image_source, file)
        image = cv2.imread(image_path,cv2.COLOR_RGB2BGR)
        image = np.array(image)
        image = compress_image(image,mutiplier)
        save_name = os.path.join('image', file)
        print("Saving {}.".format(file))
        cv2.imwrite(save_name, image)