# DeepLearningProject
Final Project for AMTH 552 for Joseph Bongo, Abhinav Bhardwaj, and Daniel Edelberg


preprocessing.py --                 crops out black space, finds largest edge and uses
                                    this to square the final image, to then resize to
                                    512x512, all in grayscale. Includes functions that 
                                    use Amazon S3 for storage.

pre_train.py --                     gets patient id, labels, combines all images to one
                                    array and saves each set of data as .npy files.
                                    Also normalizes images (subtract mean and std
                                    deviation for use in eventual network). Includes
                                    functions that use Amazon S3 for storage.

train2.ipynb --                     trains 12 of our 16 convolutional neural networks
                                    where the input is the chest x-ray image from AWS
                                    and the target output are the label 1 if the x-ray
                                    recipient has TB and 0 otherwise. includes our 
                                    winning CNN based on AlexNet.

DropoutStuff.ipynb --               trains the remaining 4 neural network architectures
                                    experimenting with different types of regularizations
                                    including L1 Penalization, L2 Penalization, and 
                                    dropout.

s3_funcs.py --                      these functions push the data in a .npy format from
                                    local machine to Amazon S3.

Deep Learning Final Report.docx --  this is the final report explaining our problem, data 
                                    data preparation steps, model results, and overall
                                    overall conclusions.
