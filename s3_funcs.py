import boto3
import os
import re

s3 = boto3.client("s3")
bucket_name = "yale-amth552-deep-learning"
mont_path = "../Downloads/MontgomerySet"
mont_dir_type = {"ClinicalReadings":".txt", "CXR_png":".png", "ManualMask":{"leftMask": ".png", "rightMask": ".png"}}

china_path = "../Downloads/ChinaSet_AllFiles"
china_dir_type = {"ClinicalReadings":".txt", "CXR_png":".png"}

def to_S3_Mont(bucket_name, path, dir_type):
    """Push Montgomery Data to S3"""
    for d1, f1 in dir_type.items():
        if type(f1) == dict:
            for d2, f2 in f1.items():
                files = [f for f in os.listdir('./' + path + "/" + d1 + "/" + d2) if re.search(f2, f)]
                for i in files:
                    s3.upload_file(path + "/" + d1 + "/" + d2 + "/" + i, bucket_name, "Images/" + d1 + "/" + d2 + "/" + i)
                    print(i)

        else:
            files = [f for f in os.listdir('./' + path + "/" + d1) if re.search(f1, f)]
            for i in files:
                s3.upload_file(path + "/" + d1 + "/" + i, bucket_name, "Images/" + d1 + "/" + i)
                print(i)

def to_S3_China(bucket_name, path, dir_type):
    """Push Shenzhen Data to S3"""
    for d1, f1 in dir_type.items():
        files = [f for f in os.listdir('./' + path + "/" + d1) if re.search(f1, f)]
        for i in files:
            s3.upload_file(path + "/" + d1 + "/" + i, bucket_name, "Images/" + d1 + "/" + i)
            print(i)

if __name__ == "__main__":
    to_S3_China(bucket_name, china_path, china_dir_type)
