from transformers import pipeline, BertTokenizer
import torch, nltk, json
from nltk.corpus import wordnet
from nltk.corpus import words

nltk.download('wordnet')
nltk.download('words')

model = pipeline('feature-extraction', model='bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embeddings = {}
corpus = words.words()

# embeddings for individual words
def get_embeddings(word):
    return model(word)[0][0]

# precompute corpus now to save time later
def precompute_embeddings(corpus):
    for word in corpus:
        embeddings[word] = get_embeddings(word)
    with open('embeddings.json', 'w') as f:
        json.dump(embeddings, f)

def load_embeddings():
    with open('embeddings.json', 'r') as f:
        embeddings = json.load(f)
    return embeddings
        
def find_words(description, corpus_embeddings):
    description_ft = get_embeddings(description)
    best_match = None
    highest_sim = -1 # cosine similarity ranges from -1 to 1
    
    for word, word_embedding in corpus_embeddings.items():
        #word_ft = get_embeddings(word)
        sim = torch.cosine_similarity(torch.tensor(description_ft), torch.tensor(word_embedding), dim=0).item()
        #print(f"Word: {word}, Similarity: {sim}")
        if sim > highest_sim:
            highest_sim = sim
            best_match = word
    return best_match

def find_def(word):
    sets = wordnet.synsets(word)
    if not sets:
        return "Ope! I couldn't find a definition for that word."
    else:
        return sets[0].definition()

def main():
    """
    Main function to load or precompute embeddings and find words based on user descriptions.
    The following steps are performed:
    1. Attempts to load precomputed embeddings (stored in embeddings.json).
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
        print("The best matches are: ", best_match)
        print("\nWould you like to see the definition of the word? (y/n)")
        if input() == 'y':
            print(find_def(best_match))
        else:
            print("\nFantastic! Would you like to search for another word? (y/n)")
            cont = input()
            if cont == 'n':
                break
            else:
                continue

if __name__ == "__main__":
    main()