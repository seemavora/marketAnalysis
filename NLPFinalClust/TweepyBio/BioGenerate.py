import tweepy 
import config
import pandas as pd

auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
api = tweepy.API(auth)

# stanford_tweets = api.user_timeline('stanford')
# for tweet in stanford_tweets:
#     print( tweet.created_at, tweet.text)

# search_words= '#springbreak' or 'college' or 'study' #Isita
# search_words = 'graduation' #Seema
# search_words = 'college life' and 'school' #Isita
# tweets = tweepy.Cursor(api.search, q = search_words, lang = 'en').items(100)
tweets = api.search( q = search_words, lang = 'en', count = '100')
userInfo = [[tweet.user.screen_name, tweet.user.description] for tweet in tweets]
tweet_text = pd.DataFrame(data = userInfo, columns = ['user','bio'])
tweet_text.drop_duplicates(subset='user')
tweet_text.to_csv('twitterBios2.csv')
print(tweet_text)