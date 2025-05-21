# AI-CHATBOT-WITH-NLP

COMPANY : CODTECH IT SOLUTION

NAME : patel bhavy

INTERN ID : CT04DL1286

DOMAIN : PYTHON PROGRAMMING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

# DESCRIPTION OF TASK

The Python script above is a basic yet functional rule-based and retrieval-based chatbot, designed using Natural Language Processing (NLP) techniques. It is intended to serve as an internship task, where the main goal is to create a chatbot that can understand user queries and respond appropriately using a corpus of text—in this case, Shakespeare’s Hamlet.


# 1. Importing Libraries and Downloading Data
The code begins by importing essential libraries such as nltk, string, and several modules from sklearn (like TfidfVectorizer and cosine_similarity). A few warnings are suppressed to keep the output clean, particularly related to tokenizer settings in TfidfVectorizer.

Using NLTK (Natural Language Toolkit), it then downloads several resources quietly:

punkt: for sentence and word tokenization.
wordnet: required for lemmatization.
stopwords: a list of common English words that don't carry significant meaning (like "the", "is", "and").
gutenberg: a corpus of literary texts including Shakespeare’s works.


# 2. ChatBot Class Initialization
A class named ChatBot is defined to encapsulate the chatbot’s behavior and functionality.

Inside the __init__ method:

A lemmatizer is created using WordNetLemmatizer to reduce words to their base form (e.g., “running” → “run”).
A list of English stop words is initialized.
The prepare_corpus() method is called to tokenize Hamlet into individual sentences which the chatbot can later use to match and retrieve appropriate responses.
A TfidfVectorizer is created using a custom tokenizer (i.e., self.preprocess) to convert the sentences into numerical vectors for comparison.
The TF-IDF matrix is computed over the sentence tokens.
Additionally, a simple dictionary of greetings is defined. This is used for rule-based responses to common questions like “hi”, “hello”, or “what is your name”.


# 3. Corpus Preparation
The prepare_corpus() method loads the raw text of Hamlet using the gutenberg.raw() function. The text is converted to lowercase and split into sentences using sent_tokenize(). These sentences are stored in self.sent_tokens for further use.

This corpus essentially acts as the chatbot’s “knowledge base.”


# 4. Text Preprocessing
The preprocess() method plays a key role in cleaning and standardizing the text:

It tokenizes the input into individual words.
Converts all text to lowercase.
Removes stop words and punctuation.
Lemmatizes each token (i.e., converts it to its dictionary form).
This preprocessing ensures that the input and corpus are both represented in a uniform format, which improves matching accuracy.


# 5. Generating a Response
The response() method is where the core logic lies:

It first checks if the input matches any key in the greetings dictionary. If found, it returns a pre-defined response immediately.
If not a greeting, it temporarily appends the user’s input to the list of sentence tokens.
It then recomputes the TF-IDF matrix with the added user input and calculates the cosine similarity between the user input and all previous sentences.
The most similar sentence is retrieved by finding the index with the highest similarity score, as long as it is above a certain threshold (0.1).
Finally, the temporary user input is removed from the sentence list to maintain the integrity of the original corpus.
If no matching sentence is similar enough, the bot responds with: "I am sorry! I don't understand you."


# 6. Main Function: Running the Chatbot
In the main() function, the script prints a welcome message and instantiates the ChatBot class. It enters an infinite loop where it waits for user input. If the user types 'bye', 'exit', or 'quit', the chatbot politely ends the conversation.

If the user interrupts with a keyboard signal (like Ctrl+C), the chatbot catches the exception and exits gracefully.


# output

<img width="709" alt="Image" src="https://github.com/user-attachments/assets/5e603633-6111-470b-841f-811fc5770f1c" />
