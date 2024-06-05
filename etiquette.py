from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import test

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

tweets = [test.filtered_tweets[i] for i in range(10)]# Remplacez par vos tweets

for tweet in tweets:
    analysis = tb(tweet)
    if analysis.sentiment[0] > 0:
        sentiment = analysis.sentiment[0]
    elif analysis.sentiment[0] < 0:
        sentiment = analysis.sentiment[0]
    else:
        sentiment = 'neutre'
    print(f"tweet: {tweet}\nSentiment: {sentiment}\n\n"

)
