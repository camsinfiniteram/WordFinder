# Pre-trained model from: https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D
# Licensed under the Apache License 2.0
import torch
from transformers import AdamW, AutoModelForSequenceClassification, BertTokenizer
from word_finder import main, corpus

# tokenize descriptions and pair them with their corresponding labels
descriptions = ["a small, round, green fruit", "a happy, jovial person", "a devious comeback"]
labels = [1, 2, 3]
    
def main():
    model.train()
    for epoch in range(5):
        for i, data in enumerate(zip(descriptions, labels)):
            description, label = data
            # tokenize description
            inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
            # forward pass  
            outputs = model(**inputs)
            logits = outputs.logits  
            labels_tensor = torch.tensor([label]).long() 
            # is this loss
            loss_fn = torch.nn.CrossEntropyLoss() 
            loss = loss_fn(logits, labels_tensor)
            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backpropagate
            optimizer.step()
        print(f"Epoch {epoch} complete! Loss: {loss.item()}")
        
    # save model
    model.save_pretrained("../dat/model")
    tokenizer.save_pretrained("../dat/model")
    print("Model saved to ../dat/model")
    
    return model

if __name__ == "__main__":
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(corpus))
    tokenizer = BertTokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    main()