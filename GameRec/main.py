import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
import numpy as np

#loading in dataset of video games
url = 'https://raw.githubusercontent.com/JessilynCollette/Video_Game_Rec/main/games.csv'
games = pd.read_csv(url)

#remove words that don't carry much meaning (stop words), reduce words to their base form
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

games['Summary'] = games['Summary'].astype(str).apply(preprocess)

#covert video games data into vectors
vectorizer = TfidfVectorizer()
game_vectors = vectorizer.fit_transform(games['Summary'])

#get user input
user_input =  input("Enter a video game summary: ")

#convert user input into vector
user_input_vector = vectorizer.transform([preprocess(user_input)])
#calculate word frequency similarity between game summary data and user input
word_similarity = cosine_similarity(user_input_vector, game_vectors).flatten()

#get index of game summary with highest word similarity
game_index = np.argpartition(word_similarity, -5) [-5:]
game_title_results = games.loc[game_index, 'Title'] [::-1]

#output predicted game title to user
print("The game you are talking about is ", game_title_results)