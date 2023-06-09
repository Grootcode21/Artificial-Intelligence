import ntlk.classify.util
from nltk import NaiveBayesClassifier
from nltk import movie_reviews

import nltk
ntlk.download('movie_reviews')

def extract_features(word_list):
    return dic([(word, True) for word in word_list])

if name == 'main':
    # load +ve and -ve reviews
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                      'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                      'Negative') for f in negative_fileids]

# split the data into train and test (80/20)
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
print("\n Number of training data points: ", len(features_train))
print("\n Number of test data points: ", len(features_test))

# train a Naive Bayes Classifier
Classifier = NaiveBayesClassifier.train(features_train)
print("\n Accuracy of the Classifier: ", nltk.classify.util.accuracy(Classifier, features_train))

print("\n Top 10 most informative words: ")
for item in classifier.most_informative_features()[:10]:
    print(item[0])

# sample input reviews
input_reviews = [
    "It is an amazing movie",
    "This is a dull movie. I would never recommend it to anyone",
    "The cinematography is pretty great in this movies",
    "I loved the movie",
    "The direction was terrible and the story was all over the place"
]

print("\n Predictions")
for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print("Predicted_sentiment: ", pred_sentiment)
    print("Probability: ", round(probdist.prob(pred_sentiment), 2))
