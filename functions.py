from gtts import gTTS
from io import BytesIO
from transformers import pipeline
import speech_recognition as sr
import nltk
nltk.download('wordnet')

def audio_to_text(audio_file_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file_path) as source:
        # Adjust for ambient noise and record the audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

        try:
            # Use Google Web Speech API to convert audio to text
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

from io import StringIO  # Add this import for Python 3 compatibility

def important_topics(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string

    def preprocess_text(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove punctuation
        tokens = [word for word in tokens if word.isalpha()]
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stop_words]
        
        # Lemmatize the words
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
        
        return " ".join(tokens)

    def extract_topics(text):
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        
        # Vectorize the text
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([preprocessed_text])
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # Extract important topics
        important_topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10-1:-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
            important_topics.append(top_words)
        
        return important_topics

    # Decode the bytes-like object to string
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Example usage
    uploaded_text = text
    important_topics = extract_topics(uploaded_text)
    print(important_topics)
    return important_topics


def text_to_audio(text, language='en', filename='audio1.wav'):
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio to a file
    tts.save(filename)
    # Read the saved file
    with open(filename, 'rb') as file:
        audio_data = file.read()
    
    return audio_data

def text_to_audio2(text, language='en', filename='audio.wav'):
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio to a file
    tts.save(filename)
    # Read the saved file
    with open(filename, 'rb') as file:
        audio_data = file.read()
    
    return audio_data

def set_domain_message(domain):
    if domain == "NOT SPECIFIC":
        return "Upload a text file:"
    elif domain == "HEALTH":
        return "Upload a HEALTH related text file:"
    elif domain == "EDUCATION":
        return "Upload an EDUCATION related text file:"
    elif domain == "SPORTS":
        return "Upload a SPORTS related text file:"
    elif domain == "BUSINESS":
        return "Upload a BUSINESS related text file:"
    elif domain == "POLITICS":
        return "Upload a POLITICS related text file:"

def set_domain_message2(domain):
    if domain == "NOT SPECIFIC":
        return "Upload a audio file:"
    elif domain == "HEALTH":
        return "Upload a HEALTH related audio file:"
    elif domain == "EDUCATION":
        return "Upload an EDUCATION related audio file:"
    elif domain == "SPORTS":
        return "Upload a SPORTS related audio file:"
    elif domain == "BUSINESS":
        return "Upload a BUSINESS related audio file:"
    elif domain == "POLITICS":
        return "Upload a POLITICS related audio file:"


def text_summarizer(text):
    text_str = text.decode('utf-8')
    model = r'text_summarization.h5' 
    summarizer = pipeline("summarization", model= "Falconsai/text_summarization")
    summary = summarizer(text_str, max_length=230, min_length=50, do_sample=False)
    extracted_summary = summary[0]['summary_text']
    return extracted_summary

def text_summarizer2(text):
    model = r'text_summarization.h5'
    summarizer = pipeline("summarization", model= "Falconsai/text_summarization")
    summary = summarizer(text, max_length=230, min_length=50, do_sample=False)
    extracted_summary = summary[0]['summary_text']
    return extracted_summary