import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import re
import nltk
import math
from nltk.stem.porter import *
from os import getcwd, listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import bert
from bert import BertModelLayer
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from imblearn.keras import BalancedBatchGenerator,balanced_batch_generator
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from scikitplot.metrics import plot_confusion_matrix
from imblearn.under_sampling import NearMiss
FullTokenizer = bert.bert_tokenization.FullTokenizer
warnings.simplefilter(action='ignore', category=FutureWarning)


# stopwords = nltk.corpus.stopwords.words('english')

nltk.download('stopwords')
stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()



def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    RE_EMOJI = re.compile('([&]).*?(;)')
    parsed_text = RE_EMOJI.sub(r'', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

# def tokenize(tweet):
#     """Removes punctuation & excess whitespace, sets to lowercase,
#     and stems tweets. Returns a list of stemmed tokens."""
#     tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
#     #tokens = re.split("[^a-zA-Z]*", tweet.lower())
#     tokens = [stemmer.stem(t) for t in tweet.split()]
#     return tokens
#
#
# def basic_tokenize(tweet):
#     """Same as tokenize but without the stemming"""
#     tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
#     return tweet.split()
#
# def count_twitter_objs(text_string):
#     """
#     Accepts a text string and replaces:
#     1) urls with URLHERE
#     2) lots of whitespace with one instance
#     3) mentions with MENTIONHERE
#     4) hashtags with HASHTAGHERE
#     This allows us to get standardized counts of urls and mentions
#     Without caring about specific people mentioned.
#     Returns counts of urls, mentions, and hashtags.
#     """
#     space_pattern = '\s+'
#     giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
#         '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     mention_regex = '@[\w\-]+'
#     hashtag_regex = '#[\w\-]+'
#     parsed_text = re.sub(space_pattern, ' ', text_string)
#     parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
#     parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
#     parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
#     return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))
#
# def other_features_(tweet):
#     """This function takes a string and returns a list of features.
#     These include Sentiment scores, Text and Readability scores,
#     as well as Twitter specific features.
#     This is modified to only include those features in the final
#     model."""
#
#     sentiment = sentiment_analyzer.polarity_scores(tweet)
#
#     words = preprocess(tweet) #Get text only
#
#     syllables = textstat.syllable_count(words) #count syllables in words
#     num_chars = sum(len(w) for w in words) #num chars in words
#     num_chars_total = len(tweet)
#     num_terms = len(tweet.split())
#     num_words = len(words.split())
#     avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
#     num_unique_terms = len(set(words.split()))
#
#     ###Modified FK grade, where avg words per sentence is just num words/1
#     FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
#     ##Modified FRE score, where sentence fixed to 1
#     FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
#
#     twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
#     features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
#                 num_unique_terms, sentiment['compound'],
#                 twitter_objs[2], twitter_objs[1],]
#     #features = pandas.DataFrame(features)
#     return features
#
#
# def get_feature_array(tweets):
#     """Takes a list of tweets, generates features for
#     each tweet, and returns a numpy array of tweet x features"""
#     feats=[]
#     for t in tweets:
#         feats.append(other_features_(t))
#     return np.array(feats)


class HatebaseTwitter():
    def __init__(self, data_path,data_column="tweet", label_column='class'):
        # Reading the Data Frame from the Hatebase Twitter CSV File
        self.df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")
        self.data_column = data_column
        self.label_column = label_column
        bert_ckpt_dir = "gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12/"
        bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
        bert_config_file = bert_ckpt_dir + "bert_config.json"

        bert_model_dir = "2018_10_18"
        bert_model_name = "uncased_L-12_H-768_A-12"
        # os.system()
        # !mkdir -p .model .model/$bert_model_name
        #
        os.system(f"mkdir -p .model .model/{bert_model_name}")
        for fname in ["bert_config.json", "vocab.txt", "bert_model.ckpt.meta", "bert_model.ckpt.index",
                      "bert_model.ckpt.data-00000-of-00001"]:
            print(f".model/{bert_model_name}/{fname}")
            if os.path.isfile(f".model/{bert_model_name}/{fname}"):
                print(f"{fname} already exists")
            else:
                cmd = f"gsutil cp gs://bert_models/{bert_model_dir}/{bert_model_name}/{fname} .model/{bert_model_name}"
                os.system(cmd)
        self._bert_ckpt_dir = os.path.join(".model/", bert_model_name)
        self._bert_ckpt_file = os.path.join(self._bert_ckpt_dir, "bert_model.ckpt")
        self._bert_config_file = os.path.join(self._bert_ckpt_dir, "bert_config.json")

    def bert_tokenize(self, train_split=0.3, max_seq_len=1024, verbose=False):
        self._tokenizer = FullTokenizer(vocab_file=os.path.join(self._bert_ckpt_dir, "vocab.txt"))
        #self.sample_size = sample_size
        self.max_seq_len = 0
        self.df[self.data_column] = self.df[self.data_column].apply(lambda x: preprocess(x))
        X_train, X_test, y_train, y_test = train_test_split(self.df[self.data_column], self.df[self.label_column], test_size=0.3,
                                                            random_state=100, stratify=self.df[self.label_column])
        train = pd.concat([X_train, y_train], axis=1).reset_index().drop('index', axis=1)
        test = pd.concat([X_test, y_test], axis=1).reset_index().drop('index', axis=1)
        if verbose:
            print(f"Train data shape: {train.shape}")
            print(f"Train data shape: {test.shape}")
        train, test = map(lambda df: df.reindex(df[self.data_column].str.len().sort_values().index),
                          [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        if verbose: print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x])

    def _prepare(self,df):
            x, y = [], []
            with tqdm(total=df.shape[0], unit_scale=True) as pbar:
                for ndx, row in df.iterrows():
                    text, label = row[self.data_column], row[self.label_column]
                    tokens = self._tokenizer.tokenize(text)
                    tokens = ["[CLS]"] + tokens + ["[SEP]"]
                    token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                    self.max_seq_len = max(self.max_seq_len, len(token_ids))
                    x.append(token_ids)
                    y.append(int(label))
                    pbar.update()
            return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)

    def create_learning_rate_scheduler(self, max_learn_rate=5e-5,
                                       end_learn_rate=1e-7,
                                       warmup_epoch_count=10,
                                       total_epoch_count=90):

        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate * math.exp(
                    math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                                total_epoch_count - warmup_epoch_count + 1))
            return float(res)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler

    def flatten_layers(self,root_layer):
        if isinstance(root_layer, keras.layers.Layer):
            yield root_layer
        for layer in root_layer._layers:
            for sub_layer in self.flatten_layers(layer):
                yield sub_layer

    def freeze_bert_layers(self,l_bert):
        """
        Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
        """
        for layer in self.flatten_layers(l_bert):
            if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
                layer.trainable = True
            elif len(layer._layers) == 0:
                layer.trainable = False
            l_bert.embeddings_layer.trainable = False

    def create_model(self, type: str, adapter_size=None):
        """Creates a classification model."""
        self.type = type
        # adapter_size = 64  # see - arXiv:1902.00751
        if type == 'binary':
            class_count = 2
        elif type == 'multi':
            class_count = 3
        else:
            raise TypeError("Choose a proper type of classification")
        # create the bert layer
        with tf.io.gfile.GFile(self._bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = adapter_size
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32', name="input_ids")
        # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        # output         = bert([input_ids, token_type_ids])
        output = bert(input_ids)

        print("bert shape", output.shape)
        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
        cls_out = keras.layers.Dropout(0.3)(cls_out)
        logits = keras.layers.Dense(units=768, activation="relu")(cls_out)
        # logits = keras.layers.Dropout(0.3)(logits)
        # logits = keras.layers.Dense(units=256, activation="relu")(logits)
        logits = keras.layers.Dropout(0.4)(logits)
        logits = keras.layers.Dense(units=class_count, activation="softmax")(logits)

        # model = keras.Model(inputs=[input_ids , token_type_ids], outputs=logits)
        # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
        model = keras.Model(inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, self.max_seq_len))

        # load the pre-trained model weights
        load_stock_weights(bert, self._bert_ckpt_file)

        # freeze weights if adapter-BERT is used
        if adapter_size is not None:
            self.freeze_bert_layers(bert)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
                      # metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")]
                      )

        model.summary()
        self.model = model
        #return model

    def eda(self):
        # Seeing the Number of CrowdFlower Annotators that Decided On a Tweet's Label
        annot_count = self.df['count'].value_counts().to_dict()

        ## Visualizing the Annotator Count
        plt.figure(figsize=(12,12))
        plt.bar(list(annot_count.keys()), list(annot_count.values()))
        plt.xlabel('Number of CrowdFlower Annotators', size=14)
        plt.ylabel('Number of Tweets Annotated', size=14)
        plt.show()

        # Visualizing the Proportion of Tweet Classification Labels and Class Imbalance
        labels = ['Hateful', 'Offensive', 'Neither']
        class_vals = self.df['class'].value_counts().to_dict()
        class_vals = {labels[i]: class_vals[i] for i in class_vals.keys()}
        fig, ax = plt.subplots(figsize=(12,12))
        ax.pie(class_vals.values(), labels=class_vals.keys(), autopct='%1.1f%%')
        ax.axis('equal')
        plt.title("Proportion of Tweet Classes", size=14)
        plt.show()

    def classify(self,total_epoch_count = 30, warmup_epoch_count = 10):#, X, type: str, classifier: str, test_prop: float, res: None, res_method: None):

        if self.type == "binary":
            self.train_y[np.where(self.train_y == 1)] = 0
            self.train_y[np.where(self.train_y == 2)] = 1
            self.test_y[np.where(self.test_y == 1)] = 0
            self.test_y[np.where(self.test_y == 2)] = 1

        #log_dir = ".log/movie_reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        training_generator, steps_per_epoch = balanced_batch_generator(self.train_x, self.train_y,
                                                                       batch_size=48,
                                                                       random_state=100)
        #total_epoch_count = 30
        # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
        self.model.fit(training_generator,
                  epochs=total_epoch_count,
                  steps_per_epoch=steps_per_epoch,
                  # validation_split=0.1,
                  callbacks=[  # keras.callbacks.LearningRateScheduler(time_decay,verbose=1),
                      # lrate,
                      self.create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                     end_learn_rate=5e-8,
                                                     warmup_epoch_count=warmup_epoch_count,
                                                     total_epoch_count=total_epoch_count)
                      #,

                      #keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
                  #    tensorboard_callback
                  ])

        self.model.save_weights('./movie_reviews.h5', overwrite=True)
        Y_pred_probabilities = self.model.predict(self.test_x)
        Y_pred = np.argmax(Y_pred_probabilities,axis=-1)

        # Accuracy Percentage
        print(f"Accuracy is {round(accuracy_score(self.test_y, Y_pred), 2)*100}%")

        # Classification Report
        print(classification_report(Y_pred, self.test_y))

        # Matthew's Correlation Coefficient
        print(f"Matthew's Correlation Coefficient is {matthews_corrcoef(self.test_y, Y_pred)}")

        # Plots of Confusion Matrix and ROC Curve
        plot_confusion_matrix(self.test_y, Y_pred, figsize=(10, 10))
        #
        # return model

    # def features(self):
    #     tweets = self.df['tweet'].tolist()
    #     vectorizer = TfidfVectorizer(
    #         tokenizer = tokenize,
    #         preprocessor = preprocess,
    #         ngram_range = (1,3),
    #         stop_words = stopwords,
    #         use_idf = True,
    #         smooth_idf = False,
    #         norm = None,
    #         decode_error = 'replace',
    #         max_features = 10000,
    #         min_df = 5,
    #         max_df = 0.75
    #     )
    #
    #     # Constructing the TF IDF Matrix and Getting the Relevant Scores
    #     tfidf = vectorizer.fit_transform(tweets).toarray()
    #     vocab = {v:i for i,v in enumerate(vectorizer.get_feature_names())}
    #     idf_vals = vectorizer.idf_
    #     idf_dict = {i: idf_vals[i] for i in vocab.values()}
    #
    #     # Getting POS Tags for Tweets and Saving as a String
    #     tweet_tags = []
    #     for t in tweets:
    #         tokens = basic_tokenize(preprocess(t))
    #         tags = nltk.pos_tag(tokens)
    #         tag_list = [x[1] for x in tags]
    #         tag_str = " ".join(tag_list)
    #         tweet_tags.append(tag_str)
    #
    #     # Use TF IDF vectorizer to get a token matrix for the POS tags
    #     pos_vectorizer = TfidfVectorizer(
    #         tokenizer=None,
    #         lowercase=None,
    #         preprocessor=None,
    #         ngram_range = (1,3),
    #         stop_words=None,
    #         use_idf=False,
    #         smooth_idf=False,
    #         norm=None,
    #         decode_error='replace',
    #         max_features=5000,
    #         min_df=5,
    #         max_df=0.75,
    #     )
    #
    #     # Construct POS TF Matrix and Get Vocabulary Dictionary
    #     pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    #     pos_vocab = {v:i for i,v in enumerate(pos_vectorizer.get_feature_names())}
    #
    #     # Getting the Other Features
    #     other_feature_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
    #                             "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
    #                             "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]
    #
    #     feats = get_feature_array(tweets)
    #
    #     # Join Features Together
    #     M = np.concatenate([tfidf, pos, feats], axis=1)
    #
    #     # Getting a List of Variable Names
    #     variables = ['']*len(vocab)
    #     for k,v in vocab.items():
    #         variables[v] = k
    #
    #     pos_variables = ['']*len(pos_vocab)
    #     for k,v in pos_vocab.items():
    #         pos_variables[v] = k
    #
    #     self.feature_names = variables + pos_variables + other_feature_names
    #
    #     return M
    #
    # def l1_dim_reduce(self, M):
    #     df = self.df
    #     y = df['class']
    #     X = pd.DataFrame(M)
    #     dim_reduce = SelectFromModel(LogisticRegression(solver='liblinear', class_weight='balanced', C=0.04, penalty='l1'))
    #     X_ = dim_reduce.fit_transform(X, y)
    #     return X_

