import configparser
import logging
import sys
import time

import torch
from torchtext import datasets, data
from tqdm import tqdm

from fully_attentive_classifier.classifiers import SelfAttentiveClassifier
from fully_attentive_classifier.embedders import PositionalEmbedder
from playground import Encoder

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def frobenius_norm(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def train(model, identity_matrix, epoch, lossfunction, optimizer, config, iter):
    global best_val_loss
    logging.info(f"\nEpoch {epoch}")
    model.train()
    clip = float(config["gradient_clipping_threshold"])
    penalization_coeff = float(config["penalization"])
    train_loss = 0
    total_correct = 0
    total_batches = len(iter.data()) // iter.batch_size
    pbar = tqdm(total=total_batches)

    for i, batch in enumerate(iter):
        pred_logits, attention = model.forward(batch.text[0])
        loss = lossfunction(pred_logits, batch.label)

        # Calculate penalization term
        if penalization_coeff > 1e-15:
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            # We index I because of the last batch, where we need less identity matrices, than batch_size
            extra_loss = frobenius_norm(torch.bmm(attention, attentionT) - identity_matrix[:attention.size(0)])
            loss += penalization_coeff * extra_loss

        train_loss += loss.item()

        prediction = torch.max(pred_logits, 1)[1]
        total_correct += torch.sum((prediction == batch.label).float()).item()

        pbar.set_description(
            f"Loss: {train_loss / (i + 1):.2f}, Acc: {total_correct / ((i + 1) * prediction.shape[0]):.2f}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        pbar.update(1)


def evaluate(model, lossfunction, config, iter):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    total_batches = len(iter.data()) // iter.batch_size
    pbar = tqdm(total=total_batches)
    for i, batch in enumerate(iter):
        pred_logits, attention = model.forward(batch.text[0])
        loss = lossfunction(pred_logits, batch.label)
        total_loss += loss.item()
        prediction = torch.max(pred_logits, 1)[1]
        total_correct += torch.sum((prediction == batch.label)).item()
        pbar.set_description(
            f"Val loss: {total_loss / (i + 1):.2f}, Val Acc: {total_correct / ((i + 1) * prediction.shape[0]):.2f}")
        pbar.update(1)
    return total_loss / total_batches, total_correct / len(iter.data())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    opts = config["Self_Attentive_Model"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, is_target=True)

    # make splits for data
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train_data, test_data, vectors=opts["embeddings"],
                     vectors_cache=opts["vectors_cache"])
    LABEL.build_vocab(train_data, test_data, specials_first=False)

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), batch_size=int(opts['batch_size']),
        sort_key=lambda x: len(x.text[0]), device=device)

    identity_matrix = torch.eye(int(opts['ATTENTION_hops']), requires_grad=False, device=device) \
        .unsqueeze(0) \
        .expand(int(opts['batch_size']),
                int(opts['ATTENTION_hops']),
                int(opts['ATTENTION_hops']))

    lossfunction = torch.nn.CrossEntropyLoss()
    model = SelfAttentiveClassifier(opts, TEXT.vocab, classes=int(opts["classes"]),
                                    embed_klazz=PositionalEmbedder,
                                    transducer=Encoder).to(device)
    logging.info(f"Model has {count_parameters(model)} trainable parameters.")
    logging.info(f"Manual seed {torch.initial_seed()}")
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(opts["lr"]),
                                 betas=[0.9, 0.999], eps=1e-8)
    start_time = time.time()
    try:
        for epoch in range(int(opts["epochs"])):
            train(model, identity_matrix, epoch, lossfunction, optimizer, opts, train_iter)
            val_loss, val_acc = evaluate(model, lossfunction, config, test_iter)
            print(f"Epoch {epoch} Val loss: {val_loss}, Val acc: {val_acc}")
    except KeyboardInterrupt:
        logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        logging.info('-' * 120)
        logging.info('Exit from training early.')
