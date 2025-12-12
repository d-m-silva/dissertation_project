import gc
import json
import pickle
import re

import numpy as np
import pandas as pd
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split

from ..config import FederatedData


def load_glove_twitter_emb(path_glove):
    word2vectors, word2id = {}, {}
    count = 0  # counter for word ids
    with open(path_glove, "rb") as file:
        for line in file:
            word_line = line.decode().split()
            word = word_line[0]
            word2vectors[word] = np.array(word_line[1:]).astype(float)
            word2id[word] = count
            count += 1

    return word2vectors, word2id


def partition_datauser(data_users):
    partition_users = {}
    user_idx, dum = 0, []
    count = 0
    entire = []
    current_user = data_users[0]
    data_indices = data_users.index

    for data_idx, user in enumerate(data_users):

        if user != current_user:
            current_user = user

            if len(dum) > 80:
                partition_users[user_idx] = np.arange(count, count + len(dum))
                count += len(dum)
                entire.extend(dum)

                user_idx += 1

            dum = []

        dum.append(data_indices[data_idx])

    if len(dum) > 80:
        partition_users[user_idx] = np.arange(count, count + len(dum))
        entire.extend(dum)

    partition_silo = dict((i, []) for i in range(len(partition_users) // 10 + 1))

    for user in partition_users:
        partition_silo[user // 10] = np.array(list(partition_silo[user // 10]) + list(partition_users[user]))

    ratios = []
    for i in range(len(partition_silo)):
        ratios.append(len(partition_silo[i]))
    ratios = list(np.array(ratios) / sum(np.array(ratios)))

    return partition_silo, ratios, entire


def partition_unseendatauser(data_users):
    partition_users = {}
    user_idx, dum = 0, []
    count = 0
    entire = []
    current_user = data_users[0]
    data_indices = data_users.index

    for data_idx, user in enumerate(data_users):

        if user != current_user:
            current_user = user

            if 50 <= len(dum) <= 80:
                partition_users[user_idx] = np.arange(count, count + len(dum))
                count += len(dum)
                entire.extend(dum)
                user_idx += 1

            dum = []

        dum.append(data_indices[data_idx])

    if 75 <= len(dum) <= 80:
        partition_users[user_idx] = np.arange(count, count + len(dum))
        entire.extend(dum)

    partition_silo = dict((i, []) for i in range(len(partition_users) // 10 + 1))

    for user in partition_users:
        partition_silo[user // 10] = np.array(list(partition_silo[user // 10]) + list(partition_users[user]))

    ratios = []
    for i in range(len(partition_silo)):
        ratios.append(len(partition_silo[i]))

    ratios = list(np.array(ratios) / sum(np.array(ratios)))

    return partition_silo, ratios, entire


def get_sent140_dataset(path_data_train, path_data_test):
    train = pd.read_csv(path_data_train, encoding='latin-1', header=0,
                        names=["polarity", "id", "date", "query", "user", "tweet"])

    test = pd.read_csv(path_data_test, encoding='latin-1', header=0,
                       names=["polarity", "id", "date", "query", "user", "tweet"])
    # train = train[:int(len(train)/5)]
    train = pd.concat([train, test])

    # original polarity column has values: {0, 2, 4} = {negative, neutral, positive}
    # drop neutral labels in polarity column and divide by 4 to make labels binary
    train = train[train.polarity != 2]
    train.polarity = train.polarity // 4  # 0, 1

    # droppings all columns but polarity score and the tweet

    # shuffling the rows to obtain val and train subsets
    train = train.sample(frac=1).reset_index(drop=True)
    train = train.sort_values(by='user')

    user_data = train.user

    partition, ratios, entire = partition_datauser(user_data)

    partition_test, _, entire_test = partition_unseendatauser(user_data)

    train_all = train[["polarity", "tweet"]]
    train = train_all.iloc[entire]
    test = train_all.iloc[entire_test]

    return train, test, partition, partition_test


def hashtags_preprocess(x):
    s = x.group(1)
    if s.upper() == s:
        # if all text is uppercase, then tag it with <allcaps>
        return ' <hashtag> ' + s.lower() + ' <allcaps> '
    else:
        # else attempts to split words if uppercase starting words (ThisIsMyDay -> 'this is my day')
        return ' <hashtag> ' + ' '.join(re.findall('[A-Z]*[^A-Z]*', s)[:-1]).lower()


def glove_preprocess(text):
    # for tagging urls
    text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/|www\.){1}[A-Za-z0-9.\/\\]+[]*', ' <url> ', text)
    # for tagging users
    text = re.sub("\[\[User(.*)\|", ' <user> ', text)
    text = re.sub('@[^\s]+', ' <user> ', text)
    # for tagging numbers
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
    # for tagging emojis
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("<3", ' <heart> ', text)
    text = re.sub(eyes + nose + "[Dd)]", ' <smile> ', text)
    text = re.sub("[(d]" + nose + eyes, ' <smile> ', text)
    text = re.sub(eyes + nose + "p", ' <lolface> ', text)
    text = re.sub(eyes + nose + "\(", ' <sadface> ', text)
    text = re.sub("\)" + nose + eyes, ' <sadface> ', text)
    text = re.sub(eyes + nose + "[/|l*]", ' <neutralface> ', text)
    # split / from words
    text = re.sub("/", " / ", text)
    # remove punctuation
    text = re.sub('[.?!:;,()*]+', ' ', text)
    # tag and process hashtags
    text = re.sub(r'#([^\s]+)', hashtags_preprocess, text)
    # for tagging allcaps words
    text = re.sub("([^a-z0-9()<>' `\-]){2,}", allcaps_preprocess, text)
    # find elongations in words ('hellooooo' -> 'hello <elong>')
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <elong> ", text)
    return text


def normalize_text(text):
    # constants needed for normalize text functions
    non_alphas = re.compile(u'[^A-Za-z<>]+')
    cont_patterns = [('(W|w)on\'t', 'will not'), ('(C|c)an\'t', 'can not'), ('(I|i)\'m', 'i am'),
        ('(A|a)in\'t', 'is not'), ('(\w+)\'ll', '\g<1> will'), ('(\w+)n\'t', '\g<1> not'), ('(\w+)\'ve', '\g<1> have'),
        ('(\w+)\'s', '\g<1> is'), ('(\w+)\'re', '\g<1> are'), ('(\w+)\'d', '\g<1> would'), ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]
    clean = text.lower()
    clean = clean.replace('\n', ' ')
    clean = clean.replace('\t', ' ')
    clean = clean.replace('\b', ' ')
    clean = clean.replace('\r', ' ')
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return u' '.join([y for y in non_alphas.sub(' ', clean).strip().split(' ')])


def tweet_to_vec(tweet, word2vectors):
    stemmer = PorterStemmer()
    return np.mean([word2vectors.get(stemmer.stem(t), np.zeros(shape=(200,))) for t in tweet.split(" ")], 0)


def allcaps_preprocess(x):
    return x.group().lower() + ' <allcaps> '


def process_all_tweets_2_vec(df, word2vectors):
    x = np.stack(
        df.tweet.apply(glove_preprocess).apply(normalize_text).apply(lambda el: tweet_to_vec(el, word2vectors)))
    y = df.polarity.values.reshape(-1, 1)
    return x, y


def load_sent140(path_data_train, path_data_test, twitter_glove, seed, store_file=None):
    word2vectors, word2id = load_glove_twitter_emb(twitter_glove)

    train, test, partition_train, partition_test = get_sent140_dataset(path_data_train, path_data_test)

    x_train, y_train = process_all_tweets_2_vec(train, word2vectors)
    x_test, y_test = process_all_tweets_2_vec(test, word2vectors)

    tweet_dataset = dict()
    x_generalization_test = np.empty((0, x_train.shape[1]))
    y_generalization_test = np.empty((0, y_train.shape[1]))

    for i, (part_train, part_test) in enumerate(zip(partition_train.values(), partition_test.values())):
        partition_tweets = np.take(x_train, part_train, axis=0)
        partition_labels = np.take(y_train, part_train, axis=0)

        partition_tweets_test = np.take(x_test, part_test, axis=0)
        partition_labels_test = np.take(y_test, part_test, axis=0)

        x_val, x_global_test, y_val, y_global_test = train_test_split(partition_tweets_test, partition_labels_test,
                                                                      test_size=0.5, random_state=seed)

        x_generalization_test = np.concatenate((x_generalization_test, x_global_test))
        y_generalization_test = np.concatenate((y_generalization_test, y_global_test))
        tweet_dataset[str(i)] = {"x_train": partition_tweets, "x_val": x_val, "y_train": partition_labels,
                                 "y_val": y_val}

    del word2vectors, word2id
    gc.collect()

    if store_file is not None:
        with open(f"{store_file}_clients_data.json", "wb") as file:
            pickle.dump(tweet_dataset, file)

        np.savetxt(f"{store_file}_x_generalization_test.csv", x_generalization_test, delimiter=",")
        np.savetxt(f"{store_file}_y_generalization_test.csv", y_generalization_test, delimiter=",")

    data = FederatedData(clients_data=tweet_dataset, x_gen_test=x_generalization_test, y_gen_test=y_generalization_test)

    return data


def load_sent140_preprocessed(path_clients, path_x_gen_test, path_y_gen_test):
    with open(path_clients, "rb") as file:
        tweet_dataset = pickle.load(file)
    x_generalization_test = np.loadtxt(path_x_gen_test, delimiter=",")
    y_generalization_test = np.loadtxt(path_y_gen_test, delimiter=",")

    data = FederatedData(clients_data=tweet_dataset, x_gen_test=x_generalization_test, y_gen_test=y_generalization_test)

    return data
