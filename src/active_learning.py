import glob
import pickle

from bson.objectid import ObjectId
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pymongo
import pandas as pd

from .text_normalization import normalize_texts


class active_learning:
    """
    The database will be created in the mongodb at localhost
    N is the number of elements that need to be classified initially for the classification to begin
    K is the number of elements that will be manually classified in each interation
    threshold_delta is the minimum accuracy that the model should have,
    before stopping the learning process.
    """

    def __init__(
        self,
        N,
        K,
        threshold_delta,
        path_model_save,
        len_vector=2 ** 14,
        uri_mongo=None,
        DATABASE_NAME="activeLearningDb",
        COLLECTION_NAME="activeLearningCollection",
    ):
        self.round = 1
        if not uri_mongo:
            uri_mongo = "mongodb://localhost:27017/"
        self.myclient = pymongo.MongoClient(uri_mongo)
        self.mydb = self.myclient[DATABASE_NAME]
        self.doc_collection = self.mydb[COLLECTION_NAME]
        self.model_score = 0.0
        self.N = N
        self.K = K
        self.threshold_delta = threshold_delta
        self.vectorizer = HashingVectorizer(n_features=len_vector)
        self.current_model = None
        self.path_model_save = path_model_save

    def insert_raw_texts(self, csv_path):
        list_texts = glob.glob(csv_path + "/**/*.csv", recursive=True)
        for csv_f in list_texts:
            df = pd.read_csv(csv_f, usecols=["text"])
            list_dict_texts = df.to_dict("records")
            for dic_text in list_dict_texts:
                normalized_text = normalize_texts([dic_text["text"]])
                self.doc_collection.insert_one(
                    {
                        "text": dic_text["text"],
                        "vector": list(
                            self.vectorizer.fit_transform(normalized_text).toarray()[0]
                        ),
                        "class_human": -2,
                        "class_machine": -2,
                        "to_classify": 0,
                    }
                )

    def find_N_documents(self):
        documents_to_classify = []
        for doc in self.doc_collection.find({}).limit(self.N):
            documents_to_classify.append(
                {
                    "_id": doc["_id"],
                    "text": doc["text"],
                    "class": "",
                }
            )
        df_to_classify = pd.DataFrame(documents_to_classify)
        df_to_classify.to_csv(
            self.path_model_save
            + "/documents_to_classify_round_{}.csv".format(self.round),
            index=False,
        )
        self.round += 1

    def find_K_documents(self):
        X = []
        for doc in self.doc_collection.find({"class_human": -2}):
            X.append(
                (
                    self.current_model.predict_proba([doc["vector"]])[0][1],
                    doc["_id"],
                    doc["text"],
                )
            )
        if not len(X):
            return False
        documents_to_classify = []
        for prob, _id, text in X:
            if prob < 0.65 and prob > 0.35:
                documents_to_classify.append({"_id": _id, "text": text, "class": ""})
            if len(documents_to_classify) >= self.K:
                break
        df_to_classify = pd.DataFrame(documents_to_classify)
        df_to_classify.to_csv(
            self.path_model_save
            + "/documents_to_classify_round_{}.csv".format(self.round),
            index=False,
        )
        self.round += 1

    def manually_update_classification_retrain(self):
        df = pd.read_csv(
            self.path_model_save
            + "/documents_to_classify_round_{}.csv".format(self.round),
            usecols=["_id", "class"],
        )
        list_dict = df.to_dict("records")
        for dic in list_dict:
            self.doc_collection.find_one_and_update(
                {"_id": ObjectId(dic["_id"])}, {"$set": {"class_human": dic["class"]}}
            )
        self.train_model()

    def train_model(self):
        X = []
        y = []
        for doc in self.doc_collection.find({"class_human": {"$gt": -1}}):
            X.append(doc["vector"])
            y.append(doc["class_human"])
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        self.current_model = LogisticRegression()
        self.current_model.fit(X_train, y_train)
        self.model_score = self.current_model.score(X_test, y_test)

    def stop_model_check(self):
        if not self.find_K_documents or self.model_score > self.threshold_delta:
            return True
        else:
            return False

    def dump_model_score_round(self):
        pickle.dump(
            self.current_model,
            open(self.path_model_save + "/active_learning_model.pkl", "wb"),
        )
        pickle.dump(
            self.model_score,
            open(self.path_model_save + "/current_score.pkl", "wb"),
        )
        pickle.dump(
            self.round,
            open(self.path_model_save + "/current_round.pkl", "wb"),
        )
