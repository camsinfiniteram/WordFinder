import torch, nltk, json, msgpack, random, requests, re, numpy
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, BertTokenizer
from nltk.corpus import wordnet
from nltk.corpus import words

# initialize Flask app
app = Flask(__name__)
CORS(app)

# avoid duplicate downloads
try:
    wordnet.ensure_loaded()
    words.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('words')

# globals 
embeddings = {}
all_words = words.words()
"""
NOTE: a random sample of 35,000 words can be used to speed up computation.
However, this may result in a less accurate word matching system.
"""
# corpus = random.sample(all_words, 35000)
corpus = all_words
lemmatizer = nltk.WordNetLemmatizer()

model = None
tokenizer = None

# embeddings for individual words
def get_embeddings(word):
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True,max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  
        cls_embedding = last_hidden_state[0, 0, :]  # for sentence level representation
        return cls_embedding.cpu().numpy().tolist()  

# precompute corpus now to save time later
def precompute_embeddings(corpus):
    for word in corpus:
        embeddings[word] = get_embeddings(word)
    with open('../dat/embeddings.msgpack', 'wb') as f:
        msgpack.pack(embeddings, f)
    print("Embeddings saved to embeddings.msgpack")

# load precomputed embeddings (stored in msgpack format)
def load_embeddings():
    global embeddings
    with open('../dat/embeddings.msgpack', 'rb') as f:
        embeddings = msgpack.unpack(f)
    print("Embeddings loaded from embeddings.msgpack")
    print(f"Embeddings loaded: {list(embeddings.keys())[:10]}")  # for debugging
    return embeddings

def normalize_embedding(embedding):
    norm = numpy.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

# find the best matching word in the corpus
def find_words(description, corpus_embeddings):
    description_ft = get_embeddings(description)
    description_ft = normalize_embedding(description_ft)
    if description_ft is None:
        return "Say what? Our gears are a bit rusty. We didn't understand that description."
    best_match = None
    highest_sim = -1 # cosine similarity ranges from -1 to 1
    
    for word, word_embedding in corpus_embeddings.items():
        word_tensor = torch.tensor(word_embedding)
        description_tensor = torch.tensor(description_ft).unsqueeze(0)
        word_tensor = normalize_embedding(word_tensor).unsqueeze(0)
        sim = torch.cosine_similarity(word_tensor, description_tensor, dim=1).item()
        print(f"similarity between {word} and {description}: {sim}")
        if sim > highest_sim:
            highest_sim = sim
            best_match = word
    print(f"best match: {best_match}, highest sim: {highest_sim}")
    return best_match

# find the definition of a word
def find_def(word):
    base_word = lemmatizer.lemmatize(word) # reduce word to its root (e.g. "dancing" -> "dance")
    print("baseword: ", base_word)
    # load in dictionary API
    response = requests.get(f"https://en.wiktionary.org/api/rest_v1/page/definition/{base_word}")
    if response.status_code == 200:
        data = response.json()
        if 'en' in data and len(data['en']) > 0:
            definitions = data['en'][0].get('definitions', None)
            if len(definitions) > 0:
                definition_raw = definitions[0].get('definition', None)
                # remove HTML tags
                definition = re.sub(r'<[^>]*>', '', definition_raw)
                #print("definition: ", definition)
                if definition:
                    return definition
        else:
            returned = [
                "Rare as a unicorn! Sorry, we couldn't find a definition for that word.",
                "Rust in our gears! Sorry, we couldn't find a definition for that word.",
                "We're as stumped as a tree! Sorry, we couldn't find a definition for that word."
            ]
            return random.choice(returned)
    else:
        return "Call it an egg, the way our circuits are fried! Please try again later."
    
# Flask endpoints to interact with frontend     
@app.route('/api/find_word', methods=['POST'])
def api_find_word():
    data = request.json
    description = data.get('description', '')
    corpus_embeddings = load_embeddings()
    best_match = find_words(description, corpus_embeddings)
    if not best_match:
        return jsonify({'error': 'No matching word found.'}), 404
    definition = find_def(best_match)
    return jsonify({'word': best_match, 'definition': definition})


@app.route('/api/precompute_embeddings', methods=['POST'])
def api_precompute_embeddings():
    precompute_embeddings(corpus)
    return jsonify({'message': 'Embeddings precomputed and saved.'})

def main():
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("../dat/model/")
    model.eval() # for inference
    tokenizer = BertTokenizer.from_pretrained("../dat/model/")
    
    try:
        embeddings = load_embeddings()
    except FileNotFoundError:
        print("Precomputing embeddings...")
        precompute_embeddings(corpus)
        embeddings = load_embeddings()
    
# main to run program and load model
if __name__ == "__main__":
    main()
    app.run(debug=True)
