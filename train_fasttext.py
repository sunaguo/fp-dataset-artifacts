import os
import time
from tqdm import tqdm
import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser

from fasttext import BOWEncoder
from helpers import prepare_dataset_nli

def load_data():
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)

    prepare_train_dataset = prepare_eval_dataset = \
        lambda exs: prepare_dataset_nli(exs, tokenizer, 128)
    
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)

    train_dataset = dataset['train']
    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=2,
        remove_columns=train_dataset.column_names
    )

    eval_dataset = dataset['validation']
    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=2,
        remove_columns=eval_dataset.column_names
    )

    print("train data len", len(train_dataset_featurized), "val data len", len(eval_dataset_featurized))
    return train_dataset_featurized, eval_dataset_featurized


def eval_model(model, eval_dataset_featurized):
    model.eval()
    correct = 0
    val_loss = 0.

    for evbii in range(len(eval_dataset_featurized)//batch_size+1):
        batchd = eval_dataset_featurized[evbii*batch_size:(evbii+1)*batch_size]

        xs = torch.tensor(batchd["input_ids"])
#             print("xs", xs.shape)
        lens = (xs != 0).sum(1)
#             print("lens", lens.shape)
        labels = torch.tensor(batchd['label'])
#             print("labels", labels.shape)

        logits = model(xs, lens) 
#             print("logits", logits.shape)
        logprobs = lsm(logits)
#             print("logprobs", logprobs.shape)

        loss = criterion(logprobs, labels)
        val_loss += loss.item()

        preds = torch.argmax(logprobs, dim=-1)
#             print("preds", preds.shape)
        correct += (preds == labels).sum().numpy()

    accuracy = correct / len(eval_dataset_featurized)

    return val_loss, accuracy, 



if __name__ == '__main__':
    train_dataset_featurized, eval_dataset_featurized = load_data()

    vocab_size = 30522
    emb_dim = 128
    nclass = 3

    ## vocab_size, emb_dim, class_in
    model = BOWEncoder(vocab_size, emb_dim, nclass)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)   

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss()
    lsm = nn.Softmax(dim=-1)
    save_path = 'bow.pt'

    num_epochs = 1
    best_accuracy = 0
    batch_size = 32

    for epoch in range(num_epochs):
        for bii in tqdm(range(len(train_dataset_featurized)//batch_size+1)):
            batchd = train_dataset_featurized[bii*batch_size:(bii+1)*batch_size]
            
            xs = torch.tensor(batchd["input_ids"])
    #         print("xs", xs.shape)
            lens = (xs != 0).sum(1)
    #         print("lens", lens.shape)
            labels = torch.tensor(batchd['label'])
    #         print("labels", labels.shape)
            
            model.train()
            optimizer.zero_grad()
            logits = model(xs, lens) 
    #         print("logits", labels.shape)
            logprobs = lsm(logits)
    #         print("logprobs", labels.shape)
            loss = criterion(logprobs, labels)
            loss.backward()
            optimizer.step()
            
            if bii > 0 and (bii+1)*batch_size % 1000 == 0:  
                train_loss = loss.item()
                
                val_loss, accuracy = eval_model(model, eval_dataset_featurized)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save_path)

                print('Batch: {} | Train Loss: {} | Val Loss: {} | Val Accuracy: {}'.format(
                    (bii+1), train_loss, val_loss, round(accuracy, 3)
                ))

                # d = dict(steps=)

    #     print('Epoch: {} | Train Loss: {} | Val Loss: {} | Val Accuracy: {}'.format(
    #         (epoch+1), train_loss, val_loss, round(accuracy, 3)
    #     ))