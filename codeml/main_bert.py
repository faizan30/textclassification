from datetime import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.util import dump_metrics
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator

from allennlp.data.vocabulary import Vocabulary
# from codeml.predictors import SentenceClassifierPredictor
from allennlp.models.archival import archive_model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.training.trainer import Trainer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from codeml.models import BasicClassifier

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        max_pieces=100,
        do_lowercase=True,
    )

    def tokenizer(s: str):
        return bert_token_indexer.wordpiece_tokenizer(s)[:100 - 2]

    reader = StanfordSentimentTreeBankDatasetReader(
        token_indexers={'bert': bert_token_indexer})

    train_dataset = reader.read(
        'data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            top_layer_only=True,  # conserve memory
    )

    vocab=Vocabulary.from_instances(train_dataset + dev_dataset)

    word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys = True)

    embedding_dim=word_embeddings.get_output_dim()
    cnn_encoder=CnnEncoder(embedding_dim, num_filters=100,
                     ngram_filter_sizes=(2, 3, 4, 5), output_dim=256)

    model=BasicClassifier(vocab, word_embeddings, cnn_encoder)
    optimizer=optim.Adam(model.parameters())

    iterator=BucketIterator(batch_size=32, sorting_keys=[
                              ("tokens", "num_tokens")])

    iterator.index_with(vocab)

    serialization_dir='.experiments/'+datetime.now().strftime("%m_%d_%H_%M_%S")
    if not os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)

    trainer=Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=5,
                      num_epochs=10,
                      serialization_dir=serialization_dir)

    metrics=trainer.train()
    dump_metrics(os.path.join(serialization_dir,
                              "metrics.json"), metrics, log=True)


if __name__ == '__main__':
    main()
