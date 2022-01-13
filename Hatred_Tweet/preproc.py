# remove punctuation
import string
#remove emoji
import demoji
# remove stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# lemmatize
from nltk.stem import WordNetLemmatizer


def data_preproc(text):

    # remove punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        text = text.replace('—', ' ')  # Em dash not in punctuation
        text = text.replace('’', ' ')  # ’ not in punctuation

    # lower case
    text = text.lower()

    # remove numbers
    text = ''.join(word for word in text if not word.isdigit())

    # remove emoji
    dem = demoji.findall(text)
    for item in dem.keys():
        text = text.replace(item, '')

    # remove StopWords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if not word in stop_words]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text])

    return text
