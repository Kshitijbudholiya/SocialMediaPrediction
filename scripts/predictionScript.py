import torch as th
import pandas as pd
import joblib
from random import uniform
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class EngagementDataset(th.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = th.tensor(X.values, dtype=th.float32)
        self.y = th.tensor(y.values, dtype=th.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    return predicted_class  # 0: Negative, 1: Positive

def preprocess_input(text, hashtags, platform, country, tokenizer, model, label_encoder, tfidf_vectorizer):
    # Get sentiment for the text
    sentiment = get_sentiment(text, tokenizer, model)

    # Preprocess hashtags using the loaded TF-IDF vectorizer
    hashtag_matrix = tfidf_vectorizer.transform([hashtags])

    # Handle unseen platform label
    if platform not in label_encoder.classes_:
        platform_encoded = 0  # Default encoding for unseen platform
    else:
        platform_encoded = label_encoder.transform([platform])[0]

    # Handle unseen country label
    if country not in label_encoder.classes_:
        country_encoded = 0  # Default encoding for unseen country
    else:
        country_encoded = label_encoder.transform([country])[0]

    # Combine features (no follower count during prediction input)
    X = pd.DataFrame({
        'sentiment': [sentiment],
        'platform_encoded': [platform_encoded],
        'country_encoded': [country_encoded]  # Do not include followers here
    })

    # Combine the TF-IDF features (hashtags)
    X = pd.concat([X, pd.DataFrame(hashtag_matrix.toarray())], axis=1)

    return X

def predict(text, hashtags, platform, country, followers):
    # Load saved models and encoders
    sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    label_encoder = joblib.load('./model/label_encoder.joblib')
    tfidf_vectorizer = joblib.load('./model/tfidf_vectorizer.joblib')

    # Dynamically determine input size (no follower count used for prediction input)
    input_size = len(tfidf_vectorizer.get_feature_names_out()) + 3  # +3 for sentiment, platform, country

    nn_model = th.nn.Sequential(
        th.nn.Linear(input_size, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, 2)  # Output: Retweets and Likes
    )

    # Prepare input features
    X_input = preprocess_input(text, hashtags, platform, country, tokenizer, sentiment_model, label_encoder, tfidf_vectorizer)

    # Convert input to tensor
    dataset = EngagementDataset(X_input, pd.DataFrame({'Retweets': [0], 'Likes': [0]}))  # Dummy targets
    X_tensor = dataset[0][0].unsqueeze(0)  # Add batch dimension

    # Load the saved neural network model weights
    nn_model.load_state_dict(th.load('./model/engagement_predictor_nn.pth'))
    nn_model.eval()

    # Make the prediction
    with th.no_grad():
        output = nn_model(X_tensor)

    # Convert output to numpy and return predicted values for Retweets and Likes
    predicted_values = output.numpy().flatten()
    predicted_retweets = predicted_values[0]
    predicted_likes = predicted_values[1]

    # Adjust predictions based on follower count using a random scaling factor
    retweet_adjustment = uniform(0.1, 0.3) * (followers / 100)  # Random adjustment factor for retweets based on followers
    like_adjustment = uniform(0.1, 0.3) * (followers / 100)  # Random adjustment factor for likes based on followers

    adjusted_retweets = predicted_retweets * (1 + retweet_adjustment)
    adjusted_likes = predicted_likes * (1 + like_adjustment)

    # Predict sentiment of the text (0 = Negative, 1 = Positive)
    sentiment = get_sentiment(text, tokenizer, sentiment_model)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"

    # Return the adjusted predictions and sentiment
    return adjusted_retweets, adjusted_likes, sentiment_label

def predictionOutput(text, hashtags, platform, country, followers):

    predicted_retweets, predicted_likes, sentiment_label = predict(text, hashtags, platform, country, followers)

    return predicted_retweets, predicted_likes, sentiment_label