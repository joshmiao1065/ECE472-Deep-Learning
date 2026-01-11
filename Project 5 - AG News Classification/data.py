from datasets import load_dataset
from transformers import DistilBertTokenizerFast
import numpy as np
import structlog

log = structlog.get_logger() 

def prepare_datasets(settings):
    """
    load and tokenize ag news using hugging face and distilbert
    create train val test splits
    Load and tokenize the AG News dataset and return batching helpers.
    """
    ds = load_dataset("ag_news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    max_len = settings.data.max_length

    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_len)

    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'label'])
    log.info("Datasets tokenized", train_size=len(tokenized["train"]), test_size=len(tokenized["test"]))

    # Create a small validation split from the training set (10%)
    split = tokenized["train"].train_test_split(test_size=0.1, seed=472)
    train = split["train"]
    val = split["test"]
    test = tokenized["test"]
    log.info("Train/Val/Test splits created", train_size=len(train), val_size=len(val), test_size=len(test))

    def token_iterator(split, batch_size, shuffle=True):
        """
        convert hugging face dataset into batches of numpy dicts that the training function expects
        i have input ID f, attention makss(so paddubg tokens remain indpendent to real tokens), and labels
        methodology is build index array for the split, shuffle, select examples, convert to array, and yield the batch dictionary until the split is exhausted
        EZ PZ
        """
        n = len(split)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start+batch_size]
            batch = split.select(batch_idx)
            yield {
                'input_ids': np.array(batch['input_ids'], dtype=np.int32),
                'attention_mask': np.array(batch['attention_mask'], dtype=np.int32),
                'labels': np.array(batch['label'], dtype=np.int32)
            }
    log.info("Token iterator created", batch_size=settings.training.batch_size)
    return train, val, test, token_iterator, tokenizer

    log.info("data preprocessed ")