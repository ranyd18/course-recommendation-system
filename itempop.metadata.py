# Author: Harshdeep Gupta
# Date: 02 October, 2018
# Description: Implements the item popularity model for recommendations, also uses metadata on age and gender


# Workspace imports
from evaluate import evaluate_model
from utils import test, plot_statistics
from Dataset import JobRecommenderDataset
from output_utils import get_output

# Python imports
import argparse
from time import time
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description="Run ItemPop")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pilot.jobrec.coldstart',
                        help='Choose a dataset.')
    parser.add_argument('--num_neg_test', type=int, default=499,
                        help='Number of negative instances to pair with a positive instance while testing')

    return parser.parse_args()


class ItemPop_metadata():
    def __init__(self, train_interaction_matrix: sp.dok_matrix, full_dataset: JobRecommenderDataset):
        """
        Simple popularity based recommender system
        """
        self.__alias__ = "Item Popularity with metadata"

        self.item_ratings = {}

        self.full_dataset = full_dataset

        # for each user, get his/her metadata, and put the rating in appropriate bucket

        num_users, num_items = train_interaction_matrix.shape

        for (userid, itemid ) in train_interaction_matrix.keys():
            user_age_category = full_dataset.user_metadata.get_user_age_catgory(userid)
            user_gender = full_dataset.user_metadata.get_user_gender(userid)
            user_data_key = (user_age_category, user_gender)
            if user_data_key not in self.item_ratings:
                self.item_ratings[user_data_key] =  sp.dok_matrix((num_users, num_items), dtype = np.float32)
            
            self.item_ratings[user_data_key][userid,itemid] = 1.0
               
        for key, value in self.item_ratings.items():
            # Sum the occurences of each item to get its popularity, convert to array and
            # lose the extra dimension
            dok_matrix = self.item_ratings[key]
            self.item_ratings[key] = np.array(dok_matrix.sum(axis = 0)).flatten()


    def forward(self):
        pass

    def predict(self, feeddict) -> np.array:
        # returns the prediction score for each (user,item) pair in the input
        output_scores = []
        users = feeddict['user_id']
        items = feeddict['item_id']
        for (user,item) in zip(users, items):
            user_age_category = self.full_dataset.user_metadata.get_user_age_catgory(user)
            user_gender = self.full_dataset.user_metadata.get_user_gender(user)
            user_data_key = (user_age_category, user_gender)
            output_scores.append(self.item_ratings[user_data_key][item])
        return np.array(output_scores)

    def get_alias(self):
        return self.__alias__


def main():
    args = parse_args()
    path = args.path
    dataset = args.dataset
    num_negatives_test = args.num_neg_test
    print("Model arguments: %s " % (args))

    topK = 400

    # Load data

    t1 = time()
    full_dataset = JobRecommenderDataset(
        path + dataset, num_negatives_test=num_negatives_test,
        user_attr_file=path + "user_attr.txt", item_attr_file=path + "course_attr.txt")
    train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    model = ItemPop_metadata(train, full_dataset)
    hr, ndcg, ranklist = test(model, full_dataset, topK)
    print(ranklist)
    print(hr, ndcg)
    #get_output(full_dataset, ranklist)

if __name__ == "__main__":
    main()
