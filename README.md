# DeepLearningProject
Final Project for AMTH 552 for Joseph Bongo, Abhinav Bhardwaj, and Daniel Edelberg


preprocessing.py -- crops out black space, finds largest edge and uses
                    this to square the final image, to then resize to
                    512x512, all in grayscale

pre_train.py --     gets patient id, labels, combines all images to one
                    array and saves each set of data as .npy files.
                    Also normalizes images (subtract mean and std
                    deviation for use in eventual network).


TODO:               Build the network. Proposed current model is
                    Conv Block (Conv layer, maybe 2 in a row. I've also
                    read about using "parallel" layers that then join up
                    in the pooling layer by an average), followed by
                    pool layer. Repeat this block/pool/relu a couple of
                    times. Then go into an FC layer -> softmax or some
                    log prob output. Can change up the pooling/conv
                    layers as necessary, there's a lot of stuff that can 
                    be done.
