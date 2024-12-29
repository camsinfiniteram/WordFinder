import torch, nltk, json, msgpack, random, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModel, BertTokenizer
from nltk.corpus import wordnet
from nltk.corpus import words

app = Flask(__name__)
CORS(app)

# avoid duplicate downloads
try:
    wordnet.ensure_loaded()
    words.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('words')

embeddings = {}
all_words = words.words()
corpus = random.sample(all_words, 40000) # sample for faster computation
lemmatizer = nltk.WordNetLemmatizer()


# embeddings for individual words
def get_embeddings(word):
    inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True,max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)[0].cpu().numpy()
    return embeddings.tolist()

# precompute corpus now to save time later
def precompute_embeddings(corpus):
    for word in corpus:
        embeddings[word] = get_embeddings(word)
    with open('../dat/embeddings.msgpack', 'wb') as f:
        msgpack.pack(embeddings, f)
    print("Embeddings saved to embeddings.msgpack")

def load_embeddings():
    global embeddings
    with open('../dat/embeddings.msgpack', 'rb') as f:
        embeddings = msgpack.unpack(f)
    print("Embeddings loaded from embeddings.msgpack")
    return embeddings
        
def find_words(description, corpus_embeddings):
    description_ft = get_embeddings(description)
    if description_ft is None:
        return "Say what? Our gears are a bit rusty. We didn't understand that description."
    best_match = None
    highest_sim = -1 # cosine similarity ranges from -1 to 1
    
    for word, word_embedding in corpus_embeddings.items():
        word_tensor = torch.tensor(word_embedding)
        description_tensor = torch.tensor(description_ft)
        sim = torch.cosine_similarity(word_tensor, description_tensor, dim=0).item()
        if sim > highest_sim:
            highest_sim = sim
            best_match = word
    return best_match

def find_def(word):
    base_word = lemmatizer.lemmatize(word) # reduce word to its root (e.g. "dancing" -> "dance")
    print("baseword: ", base_word)
    # load in dictionary API
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{base_word}")
    if response.status_code == 200:
        data = response.json()
        definition = data[0]['meanings'][0]['definitions'][0]['definition']
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

if __name__ == "__main__":
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    try:
        embeddings = load_embeddings()
    except FileNotFoundError:
        print("Precomputing embeddings...")
        precompute_embeddings(corpus)
        embeddings = load_embeddings()
    app.run(debug=True)
