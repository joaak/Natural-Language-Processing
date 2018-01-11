import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


song_info = pd.read_csv("songdata.csv")

sia = SentimentIntensityAnalyzer()

lyrics_score_dict = {}

for lyrics in song_info['text']:
    score = sia.polarity_scores(lyrics)  # polarity_scores returns a dictionary with the parameter and its compound, positive, negative and neutral value scores
    lyrics_score_dict[lyrics] = (score['compound'])

song_info['compound score'] = song_info['text'].map(lyrics_score_dict)

# OVERALL SONG SENTIMENT BY ARTIST #

song_info.groupby('artist', as_index = False).agg({'compound score' : 'mean'}).sort_values(by = 'compound score')

# AVERAGE: We can see that Slayer has the most negative emotion in songs and X-Treme has the most positive emotion in songs

song_info.groupby('artist', as_index = False).agg({'compound score' : 'sum'}).sort_values(by = 'compound score')

# SUM: From the total scores for each artist, we can see that Slayer has the least score (most negative emotion in songs) and Hillsong has the highest score (most positive emotion in songs)

# SENTIMENT OF INDIVIDUAL SONGS #

song_info.sort_values('compound score')[['song', 'artist','compound score']]

# MOST NEGATIVE: 'Scumbag' by Yoko Ono
# MOST POSITIVE: 'Ready' by Cat Stevens

# PLOTTING THE MOST COMMON WORDS IN TOP NEGATIVE AND POSITIVE SONGS #

lyrics = song_info.sort_values('compound score')[['song', 'text', 'compound score']]

s = ""
for i in range(len(lyrics)):
    if lyrics['compound score'][i] > 0.9995:
        s = s + lyrics['text'][i]
pos = word_tokenize(s)

t = ""
for i in range(len(lyrics)):
    if lyrics['compound score'][i] < -0.9995:
        t = t + lyrics['text'][i]
neg = word_tokenize(t)

clean_p = pos[:]
p_symbols = {'(', ')', '.', ',', '\'', '\'s', '\'ll', 'n\'t', '\'re', '\'m'} # some symbols selected to be removed from the first output after only removing stop words

for word in pos:
    if word in stopwords.words('english'):
        clean_p.remove(word)
    elif word in p_symbols:
        clean_p.remove(word)

p = nltk.FreqDist(clean_p)
p.plot(20, cumulative=False)

# As we can see, the most common word in all of the highest rated positive songs is 'Love'. Other common words apart from 'I', 'you' and other such general words are: 'like', 'yeah', 'good', 'need', 'know', 'beautiful', 'free'.

clean_n = neg[:]
n_symbols = {'(', ')', '.', ',', '\'', '\'s', '\'ll', 'n\'t', '\'re', '\'m', '?', '!', '\"'} # some symbols selected to be removed from the first output after only removing stop words

# note: '?' and '!' appeared in the most frequent characters in songs with a negative sentiment but not in those with a positive sentiment

for word in neg:
    if word in stopwords.words('english'):
        clean_n.remove(word)
    elif word in n_symbols:
        clean_n.remove(word)

n = nltk.FreqDist(clean_n)
n.plot(20, cumulative=False)

# As we can see, the most common word in all of the highest rated negative songs is 'I'. Other common words include: 'shit', 'block', 'fuck', kill', 'haters', and also 'love'.
