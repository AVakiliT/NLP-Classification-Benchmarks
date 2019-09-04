import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertModel, DistilBertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET = 'agnews'
# MAX_LEN = 60
# N_EPOCHS = 18
# NUM_CLASSES = 4
# BATCH_SIZE=32

DATASET = 'ng20'
MAX_LEN = 200
N_EPOCHS = 18
NUM_CLASSES = 20
BATCH_SIZE=16
#
# DATASET = 'yelp_full'
# MAX_LEN = 200
# N_EPOCHS = 4
# NUM_CLASSES = 5

#%%

df = pd.read_csv(DATASET + '/train_clean.csv')

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype('int64')

attention_matix = (token_ids_matrix != 0).astype('float')

train_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix), torch.tensor(np.array(df.label)))

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


#%%
df = pd.read_csv(DATASET + '/test_clean.csv')

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype('int64')

attention_matix = (token_ids_matrix != 0).astype('float')

test_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix), torch.tensor(np.array(df.label)))

test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

#%%

# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=NUM_CLASSES)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASSES)
model = model.cuda()

#%%
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


optimizer = AdamW(model.parameters(), lr=2e-5)

#%%
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 6

# trange is a tqdm wrapper around the normal python range
for ep in range(epochs):
    #   print('EPOCH', ep)

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    acc = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    p_bar =tqdm(train_data_loader)
    for step, batch in enumerate(p_bar):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        acc += logits.argmax(1).eq(b_labels).long().sum().item()
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1


        p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(tr_loss / nb_tr_steps, acc / nb_tr_examples))

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    #   print('EPOCH', ep)

    # Training

    with torch.no_grad():
        # Set our model to testing mode (as opposed to evaluation mode)
        model.eval()

        # Tracking variables
        ts_loss = 0
        acc = 0
        nb_ts_examples, nb_ts_steps = 0, 0


        # test the data for one epoch
        p_bar = tqdm(test_data_loader)
        for step, batch in enumerate(p_bar):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            # Forward pass
            loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            acc += logits.argmax(1).eq(b_labels).long().sum().item()


            # Update tracking variables
            ts_loss += loss.item()
            nb_ts_examples += b_input_ids.size(0)
            nb_ts_steps += 1

            p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(ts_loss / nb_ts_steps, acc / nb_ts_examples))

        print("test loss: {}".format(ts_loss / nb_ts_steps))