import os
import numpy as np
from imageio import imread, imsave
from tqdm import tqdm 

def prepare(imgdir, outputdir):

    files = sorted(os.listdir(imgdir))
    num = len(files)

    names = []
    labels = []
    images = []

    for file in tqdm(files):

        input_path = os.path.join(imgdir, file)

        # Each file is labeled for TB if file has 1 at the end
        # else clear is 0
        filename = os.path.splitext(file)[0]
        elements = filename.split('_')
        name = elements[1]
        label = int(elements[2])

        img = imread(input_path, as_gray=True, )

        labels.append(label)
        names.append(name)
        images.append(img)

    # Make single matrix
    images = np.stack(images)
    labels = np.array(labels)

    # Normalize images
    images -= np.mean(images)
    images /= np.std(images)

    images = images.reshape((images.shape[0], images.shape[1],
                            images.shape[2], 1))

    np.save(outputdir + '_images.npy', images)
    np.save(outputdir + '_labels.npy', labels)
    np.save(outputdir + '_patientid.npy', names)

if __name__ == '__main__':
    prepare('Images/Cropped', 'full')
