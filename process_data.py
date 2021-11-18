import nltk


def convert_sentiment(csv):
    sentiment = csv['sentiment']
    new_csv = csv[csv.sentiment != 'Neutral']
    for x in range(len(new_csv['sentiment'])):
        if new_csv['sentiment'][x] == 'Extremely Positive' or new_csv['sentiment'][x] == 'Positive':
            new_csv['sentiment'][x] = 1
        elif new_csv['sentiment'][x] == 'Negative' or new_csv['sentiment'][x] == 'Extremely Negative':
            new_csv['sentiment'][x] = 0
    return new_csv


def pre_process(line):
    new_words = []
    lem = nltk.WordNetLemmatizer()
    for word in line.split():
        if not (word == '' or word.startswith('http') or word.startswith('www') or word.startswith('@')):
            if word.startswith('#'):
                word = word[1:]
            word = ''.join(e for e in word if e.isalnum())
            word = word.lower()
            if word != '':
                word = lem.lemmatize(word)
                new_words.append(word)
    return bytes(' '.join(new_words), 'utf-8')

