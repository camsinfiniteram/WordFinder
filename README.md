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

### Acknowledgments:
This project utilizes the Wikimedia REST API to fetch word definitions. For more information on the API, please refer to their documentation [here](https://en.wiktionary.org/api/rest_v1/).

This project also uses a pre-trained model from [TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D), based on the original paper:

Jiao, Xiaoqi, et al. "TinyBERT: Distilling BERT for Natural Language Understanding", *arXiv preprint arXiv:1909.10351*, 2019.

### Known Issues/Areas for Improvement:
- Lemmatization is not perfect. Some words may not be fully reduced to their base forms, and the dictionary may not recognize them.
