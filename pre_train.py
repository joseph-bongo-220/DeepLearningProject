import os
import numpy as np
from imageio import imread, imsave
from tqdm import tqdm 

import boto3
import re

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

        img = imread(input_path, as_gray=True)

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

def get_s3_keys(client, bucket, prefix, file_type = "png"):
    """Get a list of keys in an S3 bucket."""
    keys = []
    resp = client.list_objects(Bucket = bucket, Prefix = prefix+'/', Delimiter='/')
    for obj in resp['Contents']:
        if re.search("[.]" + file_type + "$", obj['Key']):
            keys.append(obj['Key'])
    return keys

def prepare_s3(imgdir, outputdir, bucket, file_type = "png"):
    s3 = boto3.client("s3")

    files = get_s3_keys(s3, bucket, imgdir)
    num = len(files)

    names = []
    labels = []
    images = []

    for file in tqdm(files):
        # Each file is labeled for TB if file has 1 at the end
        # else clear is 0
        filename = re.findall("(?<=/).*(?=[.]" + file_type + ")", file)[0]
        s3.download_file(bucket, file, filename + "." + file_type)

        elements = filename.split('_')
        name = elements[1]
        label = int(elements[2])

        img = imread(filename + "." + file_type, as_gray=True)

        labels.append(label)
        names.append(name)
        images.append(img)

        os.remove(filename + "." + file_type)

    # Make single matrix
    images = np.stack(images)
    labels = np.array(labels)

    # Normalize images
    images -= np.mean(images)
    images /= np.std(images)

    images = images.reshape((images.shape[0], images.shape[1],
                            images.shape[2], 1))

    np.save(outputdir + '_images.npy', images)
    s3.upload_file(outputdir + '_images.npy', bucket, outputdir + '_images.npy')
    os.remove(outputdir + '_images.npy')

    np.save(outputdir + '_labels.npy', labels)
    s3.upload_file(outputdir + '_labels.npy', bucket, outputdir + '_labels.npy')
    os.remove(outputdir + '_labels.npy')

    np.save(outputdir + '_patientid.npy', names)
    s3.upload_file(outputdir + '_patientid.npy', bucket, outputdir + '_patientid.npy')
    os.remove(outputdir + '_patientid.npy')

if __name__ == '__main__':
    prepare_s3('Cropped', 'full', "yale-amth552-deep-learning")
