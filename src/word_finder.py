import torch, nltk, json, msgpack, random, requests
from transformers import AutoModel, BertTokenizer
from nltk.corpus import wordnet
from nltk.corpus import words

# avoid duplicate downloads
try:
    wordnet.ensure_loaded()
    words.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('words')

embeddings = {}
all_words = words.words()
corpus = random.sample(all_words, 50000) # sample for faster computation
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
        elif definition == "":
            return "Rare as a unicorn! We couldn't find a definition for that word."
    else:
        return "Rust in our gears! Sorry, we couldn't load the definition for that word."

def main():
    """
    Main function to load or precompute embeddings and find words based on user descriptions.
    The following steps are performed:
    1. Attempts to load precomputed embeddings (stored in embeddings.msgpack).
    2. If embeddings are not found, precompute and load them.
    3. Enters an interactive loop where the user can input a description of a word.
    4. Searches and displays the best matching word based on the description.
    5. Prompts the user if they want to see the word's definition.
    6. Asks the user if they want to continue or exit.
    Exceptions:
        FileNotFoundError: If the embeddings file is not found, it triggers precomputation of embeddings.
    Returns:
        None
    """
    try:
        print("Loading data...hang tight!")
        corpus_embeddings = load_embeddings()
    except FileNotFoundError:
        print("Precomputing...this may take a while.")
        precompute_embeddings(corpus)
        print("Embeddings precomputed!")
        corpus_embeddings = load_embeddings()
            
    while True:
        print("Input a description of the word you are looking for:")
        desc = input()
        best_match = find_words(description=desc, corpus_embeddings=corpus_embeddings)
        if best_match is None:
            print("Call it an egg, the way our circuits are fried! We can't find a word to match that description.")
        print("The best matches are: ", best_match)
        print("\nWould you like to see the definition of the word? (y/n)")
        if input() == 'y':
            print(find_def(best_match))
        else:
            print("\nWould you like to search for another word? (y/n)")
            cont = input()
            if cont == 'n':
                print("Thanks for stopping by! We hope you found what was on the tip of your tongue.")
                break
            else:
                continue

if __name__ == "__main__":
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    main()