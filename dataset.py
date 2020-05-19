from PIL import Image
import json
import os
from skimage import transform
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

"""
This file preprocesses the Yelp photo and star rating data, 
declares the CustomDataset class to be used in a pytorch
DataLoader, and provides some utils. 
"""

def pad_to_size(np_im, pad_size):
    """
    Accepts a numpy array of shape (C, H, W), and pads it to be
    (C, pad_size[0], pad_size[1]). It is assumed that 
    H, W <= pad_size[0], pad_size[1]
    """
    _, h, w = np_im.shape

    pad_hl = (pad_size[0] - h) // 2
    pad_hr = (pad_size[0] - h) // 2
    
    # to allows for odd-length pads
    if (pad_size[0] - h) % 2:
        pad_hr += 1

    pad_wl = (pad_size[1] - w) // 2
    pad_wr = (pad_size[1] - w) // 2

    if (pad_size[1] - w) % 2:
        pad_wr += 1

    return np.pad(np_im, [(0, 0), (pad_hl, pad_hr), (pad_wl, pad_wr)])


def generate_ids2photos(photos_json_path):
    """
    Accepts the path to the photos.json file, uses that to construct
    a doubly nested dictionary from label -> (business_id -> jpg file)
    which is returned. 
    """
    raw_photo_data = open(photos_json_path, 'r')
    photo_data = raw_photo_data.readlines()

    ids2photos = {}

    for line in photo_data:
        data = json.loads(line)
        b_id = data['business_id']
        p_id = data['photo_id']
        label = data['label']

        if label not in ids2photos:
            ids2photos[label] = defaultdict(list)

        ids2photos[label][b_id].append(p_id + '.jpg')

    return ids2photos


def generate_ids2stars(bus_path):
    """ 
    Accepts the file path to the business.json, uses that to construct
    a dictionary from business ids -> star ratings which is returned.
    """

    raw_ratings_data = open(bus_path, 'r')
    ratings_data = raw_ratings_data.readlines()
    
    ids2stars = defaultdict(int)
    for line in ratings_data:
        d = json.loads(line)
        ids2stars[d['business_id']] = d['stars']

    return ids2stars


def generate_dset_sizes(photo_count):
    """
    Accepts int photo_count and uses it to define the size of the 
    .hdf5 arrays to be created.
    Sets train as 90% of data, with val and test at 5% of the data. 
    Called by: combine_photos_ratings
    """
    train_size = 9 * photo_count // 10 # 90%
    val_size = photo_count // 20 # 5% 
    test_size = photo_count // 20 + photo_count % 20 # 5%

    boundaries = [train_size, train_size + val_size]

    return train_size, val_size, test_size, boundaries


def jpg2np(photos_dir, photo_name, pad_size, reduction=(1,3,3)):
    """
    Uses the photo direction and .jpg name to open an image
    file, convert it to numpy, and then returns that image. 
    Shrinks the image by a factor of 2
    """
    jpg_file = photos_dir + photo_name
    im = Image.open(jpg_file)

    np_im = np.array(im)    
    np_im = np.transpose(np_im, (2, 0, 1)) #reshape for torch

    # pad to a consistent size
    if pad_size:
        np_im = pad_to_size(np_im, pad_size)

    return np_im[::reduction[0], ::reduction[1], ::reduction[2]]


def generate_star_histograms(bus_path, label):
    """
    Generates a histogram of star ratings for all the businesses included 

    """



def combine_photos_and_ratings(bus_path, photos_json_path, photos_dir, processed_path, 
        override=False, pad_size=None, verbose=False):
    """
    The first step of preprocessing, accesses the business.json to acquire business
    IDs and ratings, uses those business ID to find associated images in photos.json. 
    Find .jpg name in photos.json and opens the associated jpg. That jpg is converted
    to an np array, padded to a standard sixe, stored with it's rating, and saved as 
    a .npy file.

    @param bus_path (string): path to business.json
    @param photos_json_path (string): path to photos.json
    @param photos_dir (string): directory containing .jpg photos
    @param processed_path (string): location where processed data is saved
    @param override (bool): flag to control rewriting the already preprocessed file
    @param pad_size (tuple): Tuple containing max height and width for an image

    """
    if os.path.exists(processed_path + '_food' + '_inputs.npy') and not override:
        print("Preprocessed data file already exists")
        return
    
    # Associate each business ID with all it's photos and ratings
    ids2photos = generate_ids2photos(photos_json_path)
    ids2stars = generate_ids2stars(bus_path)
    print(len(ids2photos))
    print(len(ids2stars))
    
    # we now want to segregate by labels, get the amount of pictures associated with each labels
    labels = ids2photos.keys()
    for label in labels:
        print('\nBeginning processing {} data'.format(label))
        print('-' * 15)

        businesses = ids2stars.keys()
        data = ids2photos[label]

        photo_count = sum([len(data[b_id]) for b_id in businesses])
        train_size, val_size, test_size, boundaries = generate_dset_sizes(photo_count)
        print(train_size, val_size, test_size)
        
        inputs = np.empty((photo_count, 3, 134, 200), dtype="int8")
        outputs = np.empty((photo_count,), dtype="int8")


        # declare .hdf5 file and associated Datasets
#       f = h5py.File(processed_path + '_' + label + ".hdf5", "w")
#       train_dset_x = f.create_dataset("train_x", (train_size, 3, 134, 200), dtype='int8')
#       train_dset_y = f.create_dataset("train_y", (train_size,), dtype='int8')
#       
#       val_dset_x = f.create_dataset("val_x", (val_size, 3, 134, 200), dtype='int8')
#       val_dset_y = f.create_dataset("val_y", (val_size,), dtype='int8')
#       
#       test_dset_x = f.create_dataset("test_x", (test_size, 3, 134, 200), dtype='int8')
#       test_dset_y = f.create_dataset("test_y", (test_size,), dtype='int8')
        
        example_count = 0 # to track relative index within each h5py file
        star_histogram = [0] * 10 # 10 different possible star ratings

        for b_id in businesses:
            stars = ids2stars[b_id]
            stars = int(2 * stars)
            for photo_name in data[b_id]:
                
                np_im = jpg2np(photos_dir, photo_name, pad_size)
                
                inputs[example_count] = np_im

                outputs[example_count] = stars
#               if example_count < boundaries[0]:
#                   train_dset_x[example_count] = np_im
#                   train_dset_y[example_count] = stars

#               elif example_count < boundaries[1]: 
#                   idx = example_count - boundaries[0]
#                   val_dset_x[idx] = np_im
#                   val_dset_y[idx] = stars
#               else:
#                   idx = example_count - boundaries[1]
#                   test_dset_x[idx] = np_im
#                   test_dset_y[idx] = stars
                
                example_count += 1
              
                if verbose and example_count % verbose == 0:
                    print('Processed {} images out of {}'.format(example_count, photo_count))
                    
        np.save(processed_path + '_' + label + '_inputs', inputs, allow_pickle=True)
        np.save(processed_path + '_' + label + '_outputs', outputs, allow_pickle=True)

class CustomDataset(Dataset):
    """
    To be used in the construction of a dataloader, extends the 
    __getitem__ API and the _yield_ API.
    
    Paths to different data caches should not be stored locally long-term. 


    TODO: will likely need to be modified to read and write on the VM
    """

    def __init__(self, bus_path, label="food", mode="train", 
            override=False, verbose=0):
        """
        @param bus_path (string): filename containing business data
        @param mode (string): Default "Train", "Val" and "Test" also acceptable
        @param label (string): one of "food", "inside", "outside", "food", "menu", 
                               selects which subset of yelp photo data we are 
                               training on
        @param override (bool): controls rewriting processed data files
        @param verbose (int): if set to a positive value, preprocess will print updates 
                              each time <verbose> photos have been processed
        """
        self.check_params(label, mode, verbose)    

        photos_json_path = "../all_data/data/photos.json"
        photos_dir = "../all_data/data/photos/"
        processed_path = "processed_data/combined" 

        combine_photos_and_ratings(bus_path, photos_json_path, photos_dir, 
                processed_path, override=override, pad_size=(400, 600), verbose=verbose)

        # Load the dataset from npy file
        self.dataset_x = np.load(processed_path + '_' + label + '_inputs.npy', allow_pickle=True)
        self.dataset_y = np.load(processed_path + '_' + label + '_outputs.npy', allow_pickle=True)

    def __getitem__(self, idx):
        input_, target_ = self.dataset_x[idx], self.dataset_y[idx]

        # numpy image is already correctly stored with dims (C, H, W)
        input_ = torch.from_numpy(input_)
        return input_, target_

    def __len__(self):
        return len(self.dataset_x)


    def check_params(self, label, mode, verbose):
        """
        Called by CustomDataset, ensures all flags are expected. 
        """
        if mode not in ["train", "test", "val"]:
            raise ValueError("Unaccepted dataset mode received")

        if label not in ["food", "outside", "drink", "inside", "menu"]:
            raise ValueError("Unaccepted dataset label received")

        if verbose < 0:
            raise ValueError("Verbose cannot be negative")


# uncomment this call outside of the context of the dataloader
# to preprocess data before training
bus_path = "../all_data/data/yelp_academic_dataset_business_tiny.json"
ds = CustomDataset(bus_path, override=True, verbose=5000)
