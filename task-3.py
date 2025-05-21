import warnings
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")

import nltk
import string
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required nltk data quietly
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('gutenberg', quiet=True)

# (Rest of your chatbot code follows here...)


class ChatBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sent_tokens = []
        self.prepare_corpus()

        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess, stop_words=None)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.sent_tokens)

        # Simple greetings dictionary
        self.greetings = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hi there! What can I do for you?",
            "how are you": "I am just a bot, but I'm doing great! Thanks for asking.",
            "what is your name": "I am CodTech NLP Chatbot, your friendly assistant.",
            "tell me your age": "I am timeless — created for this internship task!",
            "who are you": "I am a chatbot built using Python and NLP techniques."
        }

    def prepare_corpus(self):
        raw_text = gutenberg.raw('shakespeare-hamlet.txt').lower()
        self.sent_tokens = sent_tokenize(raw_text)

    def preprocess(self, text):
        # Tokenize, lowercase, remove punctuation & stopwords, and lemmatize
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token not in string.punctuation
        ]
        return filtered_tokens

    def response(self, user_input):
        user_input_lower = user_input.lower()

        # Check greetings dictionary first
        for key in self.greetings:
            if key in user_input_lower:
                return self.greetings[key]

        # Add user input temporarily to sentences
        self.sent_tokens.append(user_input_lower)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sent_tokens)

        user_vec = self.vectorizer.transform([user_input_lower])
        cosine_similarities = cosine_similarity(user_vec, self.tfidf_matrix[:-1])  # exclude user input itself
        max_sim_index = cosine_similarities.argmax()
        max_sim_score = cosine_similarities[0][max_sim_index]

        self.sent_tokens.pop()  # remove user input from corpus

        if max_sim_score < 0.1:
            return "I am sorry! I don't understand you."
        else:
            return self.sent_tokens[max_sim_index]

def main():
    print("Script started")
    print("Hello! I am CodTech NLP Chatbot. Type 'bye' to exit.")
    chatbot = ChatBot()
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['bye', 'exit', 'quit']:
                print("Chatbot: Goodbye! Have a nice day.")
                break
            print("Chatbot:", chatbot.response(user_input))
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye! (Interrupted by user)")


if __name__ == "__main__":
    main()
