from PIL import Image
import json
import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

def combine_photos_and_ratings(bus_path, photo_paths, photos_dir, out_path, override=False):
    """
    The first step of preprocessing, accesses the business.json to acquire business
    IDs and ratings, uses those business ID to find associated images in photos.json. 
    Find .jpg name in photos.json and opens the associated jpg. That jpg is converted
    to an np array, stored with it's rating, and saved as a .npy file.

    @param bus_path (string): path to business.json
    @param photo_paths (string): path to photos.json
    @param photos_dir (string): directory containing .jpg photos
    @param out_path (string): location where processed data is saved
    @param override (bool): flag to control rewriting the already preprocessed file

    @return max_dims (tuple): dimensions of the largest photo array
                              in the dataset, for use in future padding
    """
    if os.path.exists(out_path) and not override:
        print("Data already preprocessed to combine photos with ratings")
        return

    raw_photo_data = open(photo_paths, 'r')
    photo_data = raw_photo_data.readlines()
    
    # Associate each business ID with all it's photos
    ids2photos = defaultdict(list)
    for line in photo_data:
        data = json.loads(line)
        b_id = data['business_id']
        p_id = data['photo_id']

        ids2photos[b_id].append(p_id + '.jpg')


    # Get the star rating of each business, begin constructing numpy_array
    raw_ratings_data = open(bus_path, 'r')
    ratings_data = raw_ratings_data.readlines()

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
            photos_with_ratings.append(np.asarray([np_im, rating]))


    photos_with_ratings = np.asarray(photos_with_ratings)
    np.save(out_path, photos_with_ratings, allow_pickle=True)




bus_path = "data/yelp_academic_dataset_business.json"
photos_path = "data/photos.json"
photos_dir = "data/photos/"
out_path = "data/comined_photos_ratings" 

combine_photos_and_ratings(bus_path, photos_path, photos_dir, out_path)

