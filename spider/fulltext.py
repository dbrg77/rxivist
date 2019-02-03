import pickle
import re

import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(content):
  # Chop off the extraneous stuff at the end (references, etc.)
  cutoff = content.find('## Footnotes')
  if cutoff > -1:
    content = content[0:cutoff]

  # create stems of all the words
  # TODO: Look into lemmatizing instead?
  stemmer = SnowballStemmer('english', ignore_stopwords=True)
  content = ' '.join([stemmer.stem(x) for x in content.split(' ')])
  return content

def get_article_fulltext(spider, url):
  return # NOOP

  spider.log.record("Fetching fulltext for article", 'debug')
  try:
    resp = spider.session.get(f"{url}.full.txt")
  except Exception as e:
    spider.log.record(f"Error requesting article text. Bailing: {e}", "error")
    return None
  spider.log.record('Got content!', 'debug')
  content = resp.content.decode('utf-8')
  print(f'It is {len(content)} long')
  if len(content) < 4500:
    spider.log.record(f'Fulltext is only {len(content)} characters; assuming it\'s not processed yet.', 'info')
    return

  content = clean_text(content)

def get_abstracts(spider):
  entries = []
  print("Getting abstracts")
  with spider.connection.db.cursor() as cursor:
    cursor.execute("""
      SELECT collection, abstract
      FROM prod.articles
      WHERE collection IS NOT NULL
        AND abstract IS NOT NULL
      LIMIT 10000;
    """)
    print("Cleaning abstracts")
    for result in cursor:
      entries.append((result[0], clean_text(result[1])))
  return entries

cats = ['animal-behavior-and-cognition','biochemistry','bioengineering','bioinformatics','biophysics','cancer-biology','cell-biology','clinical-trials','developmental-biology','ecology','epidemiology','evolutionary-biology','genetics','genomics','immunology','microbiology','molecular-biology','neuroscience','paleontology','pathology','pharmacology-and-toxicology','physiology','plant-biology','scientific-communication-and-education','synthetic-biology','systems-biology','zoology']

def analyze(spider, modelfile=None, save=False):
  # create the transform
  vectorizer = CountVectorizer(
    ngram_range=(1,2), # include two-word phrases
    min_df = 3 # throw away phrases that show up in < 3 papers
  )
  if modelfile is None:
    nltk.download('stopwords')
    content = get_abstracts(spider)
    print("ANALYZING ABSTRACTS!")

    print("Encoding...")
    X = vectorizer.fit_transform([x[1] for x in content]).toarray() # just the text
    Y = np.array([cats.index(x[0]) for x in content]) # just the labels

    clf = RandomForestClassifier()
    print("Fitting...")
    clf.fit(X, Y)
    if save:
      print("Saving model...")
      with open('model.pickle', 'wb') as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
      with open('modelx.pickle', 'wb') as f:
        pickle.dump(vectorizer, f, pickle.HIGHEST_PROTOCOL) # just the text
  else:
    print(f"Loading model from {modelfile}")
    with open(f'{modelfile}.pickle', 'rb') as f:
      clf = pickle.load(f)
    with open(f'{modelfile}x.pickle', 'rb') as f:
      vectorizer = pickle.load(f)

  print("Ready.")
  while True:
    q = input('Enter abstract: ')
    if q == 'x':
      break
    answer = clf.predict_proba(vectorizer.transform([q]).toarray())
    for i in range(len(answer[0])):
      if answer[0][i] > 0:
        print(f'{cats[i]}: {answer[0][i]}')
