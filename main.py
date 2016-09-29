import tweepy
import os
import json
import time
import random
import string

from pandas import DataFrame, concat

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

consumer_token = '????'
consumer_secret = '????'
access_token = '???'
access_token_secret = '????'

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def connect():
    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

def get_tweets(username, count=200):
    api = connect()

    oldest = None
    tweets = 'empty'

    all_tweets = []

    if not os.path.exists('./%s' % username):
        os.makedirs('./%s' % username)

    while len(tweets) > 0:
        if oldest:
            tweets = api.user_timeline(
                screen_name=username, count=count, max_id=oldest
            )
        else:
            tweets = api.user_timeline(
                screen_name=username, count=count
            )

        if tweets:
            oldest = tweets[-1].id - 1

        for tweet in tweets:
            all_tweets.append(dict(time=str(tweet.created_at), tweet=tweet.text))

        time.sleep(1)

    f = open('./%s/tweets.json' % username, 'w')
    f.write(json.dumps(all_tweets))
    f.close()

    return all_tweets

def get_data_frame(username):

    data = open('./%s/tweets.json' % username)
    if data:
        tweets = json.loads(data.read())

        formated = [ dict(text=i['tweet'], classification=username) for i in tweets ]
        ids = [ '%s-%s' % (i['time'], id_generator()) for i in tweets ]
        return DataFrame(formated, index=ids)

def merge_data_frames(*args):
    return concat(args)

def predict(data_frame, example):

    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
        ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',         MultinomialNB())
    ])

    pipeline.fit(data_frame['text'].values, data_frame['classification'].values)

    res = zip(pipeline.classes_, pipeline.predict_proba([example])[0])
    for i in sorted(res, key=lambda x:x[1]):
        print i

