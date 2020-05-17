from PIL import Image
import json
import os
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


def combine_photos_and_ratings(bus_path, photos_json_path, photos_dir, out_path, 
        override=False, pad_size=None):
    """
    The first step of preprocessing, accesses the business.json to acquire business
    IDs and ratings, uses those business ID to find associated images in photos.json. 
    Find .jpg name in photos.json and opens the associated jpg. That jpg is converted
    to an np array, padded to a standard sixe, stored with it's rating, and saved as 
    a .npy file.

    @param bus_path (string): path to business.json
    @param photos_json_path (string): path to photos.json
    @param photos_dir (string): directory containing .jpg photos
    @param out_path (string): location where processed data is saved
    @param override (bool): flag to control rewriting the already preprocessed file
    @param pad_size (tuple): Tuple containing max height and width for an image

    """
    if os.path.exists(out_path + '_train' + '.npy') and not override:
        print("Preprocessed data file already exists")
        return

    raw_photo_data = open(photos_json_path, 'r')
    photo_data = raw_photo_data.readlines()
    
    # Associate each business ID with all it's photos
    ids2photos = defaultdict(list)
    for line in photo_data:
        data = json.loads(line)
        b_id = data['business_id']
        p_id = data['photo_id']
        label = data['label']

        ids2photos[(b_id, label)].append(p_id + '.jpg')

    # Get the star rating of each business, begin constructing numpy_array
    raw_ratings_data = open(bus_path, 'r')
    ratings_data = raw_ratings_data.readlines()

    labels = set([key[1] for key in ids2photos])

    for label in labels:
        photos_with_ratings = []

        repeats = set()
        count = 0
        size = len(ratings_data)
        for line in ratings_data:

            data = json.loads(line)
            b_id = data["business_id"]
            rating = float(data["stars"])

            if b_id in repeats:
                print('weird duplication of businesses in dataset?') # just to be safe

            # many businesses do not have photos in the dataset
            if b_id not in ids2photos:
                continue

            repeats.add(b_id)

            for photo_name in ids2photos[b_id]:
                jpg_file = photos_dir + photo_name
                im = Image.open(jpg_file)

                np_im = np.array(im)    
                np_im = np.transpose(np_im, (2, 0, 1))

                # pad to a consistent size
                if pad_size:
                    np_im = pad_to_size(np_im, pad_size)

                photos_with_ratings.append(np.asarray([np_im, rating]))

        photos_with_ratings = np.asarray(photos_with_ratings)
        num_data = len(photos_with_ratings)
        train_cutoff = num_data - (num_data // 10) # 90%
        val_cutoff = num_data - (num_data // 10) + (num_data // 20) # 5%

        label = '_' + label 
        # train set

        np.save(out_path + label + '_train', photos_with_ratings[:train_cutoff], allow_pickle=True)
        # val set
        np.save(out_path + label + '_val', photos_with_ratings[train_cutoff:val_cutoff], allow_pickle=True)
        # test set
        np.save(out_path + label + '_test', photos_with_ratings[val_cutoff:], allow_pickle=True)



class CustomDataset(Dataset):
    """
    To be used in the construction of a dataloader, extends the 
    __getitem__ API and the _yield_ API.
    
    Paths to different data caches should not be stored locally long-term. 
    """

    def __init__(self, bus_path, label="food", mode="train", override=False):
        """
        @param bus_path (string): filename containing business data
        @param mode (string): Default "Train", "Val" and "Test" also acceptable
        @param override (bool): controls rewriting processed data files
        """
        if mode not in ["train", "test", "val"]:
            raise ValueError("Unaccepted dataset mode received")


        if label not in ["food", "outside", "drink", "inside", "menu"]:
            raise ValueError("Unaccepted dataset label received")

        photos_json_path = "data/photos.json"
        photos_dir = "data/photos/"
        processed_path = "data/model_data/combined_photos_ratings" 

        combine_photos_and_ratings(bus_path, photos_json_path, photos_dir, 
                processed_path, override=override, pad_size=(400, 600))

        # Load the dataset from npy file
        dataset_path = processed_path + '_' + label + '_' + mode + '.npy'
        self.dataset = np.load(dataset_path, allow_pickle=True)

    def __getitem__(self, idx):
        input_, target_ = self.dataset[idx]

        # numpy image is already correctly stored with dims (C, H, W)
        input_ = torch.from_numpy(image)
        return input_, target_

    def __len__(self):
        return len(self.dataset)

    def get_vocab(self):
        return self.vocab

    def get_json_path(self):
        return self.json_path


# uncomment this call outside of the context of the dataloader
# to preprocess data before training
bus_path = "data/yelp_academic_dataset_business_tiny.json"
_ = CustomDataset(bus_path)
