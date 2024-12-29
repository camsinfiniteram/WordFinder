# What's the Word?
Have you ever had a word on the tip of your tongue but been unable to find it? 
<b>What's the Word</b> solves that issue! Simply describe the word you're looking for, and we'll give you the best match.
Curious to learn more? We can also provide a definition of that word to see how closely it matches your train of thought.

## To run:
1) Navigate into the backend/ folder. 
2) Run the following command: 
```
python3 word_finder.py
```
3) Navigate into the frontend/src/ folder.
4) Run the following commands:
```
npm install
npm run dev
```
5) Open the link that appears in your terminal (e.g: http://localhost:5173/).
6) Follow the prompts on the screen to enter your word description.
7) Get the best match and definition!

### Known Issues/Areas for Improvement:
- Lemmatization is not perfect. Some words may not be fully reduced to their base forms.
- While we scraped from the [dictionary API](https://github.com/meetDeveloper/freeDictionaryAPI), the WordNet corpus may not always match up to the words present in the API.
