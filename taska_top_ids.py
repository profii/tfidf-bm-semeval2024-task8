import pandas as pd
import numpy as np
import evaluate
import transformers

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer # TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from dataclasses import dataclass, field
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

stop_words_list = ['!', '"', '",', '".', '$', "'", '(', ')', '),', ').', ',', '-', '.', '."', '/', ':',
                   ';', '=', '?', '\\', '_', '–', '’', '▁', '▁"', '▁$', '▁(', '▁*', '▁1', '▁2', '▁3',
                   '▁4', '▁5', '▁6', '▁7', '▁8', '▁9', '▁10', '▁|', '▁||', '▁–', '▁“', '*', '”',
                   '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '▁to', '▁you',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   '▁a', '▁b', '▁c', '▁d', '▁e', '▁f', '▁g', '▁h', '▁i', '▁j', '▁k', '▁l', '▁m',
                   '▁n', '▁o', '▁p', '▁q', '▁r', '▁s', '▁t', '▁u', '▁v', '▁w', '▁x', '▁y', '▁z',]


@dataclass
class ModelConfig:
    model_path: str = "roberta-base"
    trust_remote_code = True,
    num_labels: int = 2
    num_hidden_layers: int = 18 # 12 def
    #ignore_mismatched_sizes = True, #dslim
    #classifier_dropout: float = 0.1
    #num_attention_heads: int = 16 # 12 def

    # hidden_act: str = "relu" # def:"gelu"; "relu", "silu" and "gelu_new"
    # position_embedding_type: str = "relative_key_query" # def:"absolute"; "relative_key", "relative_key_query"
    #hidden_dropout_prob: float = 0.3
    #attention_probs_dropout_prob: float = 0.25


@dataclass
class DatasetConfig:
    train_file: str = field(default=None, metadata={"help": "Path to train jsonl file"})
    dev_file: str = field(default=None, metadata={"help": "Path to dev jsonl file"})
    test_file: str = field(default=None, metadata={"help": "Path to test jsonl file"})


@dataclass
class TrainingArgsConfig(transformers.TrainingArguments):
    seed: int = 42
    output_dir: str = "experiments/"
    num_train_epochs: int = 3 # 10
    per_device_train_batch_size: int = 16 # 32
    per_device_eval_batch_size: int = 16 # 32
    auto_find_batch_size: bool = True
    logging_dir: str = "experiments/logs"
    logging_steps: int = 100
    run_name: str = 'exp'
    load_best_model_at_end: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: str = "wandb"

    weight_decay=0.01,
    learning_rate=2e-5,


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def preprocess_function_top(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["top"], truncation=True)
    

def sort_vectors(vector, n):
    # vector = vectorizer.transform(vector)#.toarray()
    vector = vector.toarray()
    # print(vector)
    dict_vector = {}
    for idx, vect in enumerate(vector[0]):
        if vect != 0: dict_vector[idx] = vect
    sorted_vector = sorted(dict_vector.items(), key=lambda x:x[1], reverse=True)[:n]

    return sorted_vector


def get_top(sorted_vector, feature_names):
    top_words = ' '.join([feature_names[idx] for idx, _ in sorted_vector])
    return top_words


def get_data(train_path, dev_path, vectorizer):
    """
    function to read dataframe with columns
    """
    N = 10

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(dev_path, lines=True)
    
    train_df.drop(['model', 'source', 'id'], axis=1, inplace=True)
    test_df.drop(['model', 'source'], axis=1, inplace=True)

    vectorizer.fit(train_df['text'].tolist())
    feature_names = vectorizer.get_feature_names_out()

    train_vector = vectorizer.transform(train_df['text'])
    top_list = []
    for idx in tqdm(range(len(train_df['text']))):
        top_list.append(get_top(sort_vectors(train_vector[idx], N), feature_names))
    del train_vector

    dev_vector = vectorizer.transform(test_df['text'])
    top_list_dev = []
    for idx in tqdm(range(len(test_df['text']))):
        top_list_dev.append(get_top(sort_vectors(dev_vector[idx], N), feature_names))
    del dev_vector, feature_names

    train_df['top'] = top_list
    test_df['top'] = top_list_dev

    train_df, dev_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

    print('Dev:', dev_df.head())

    return train_df, dev_df, test_df


def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = 'infr_project'

    parser = transformers.HfArgumentParser(
        (ModelConfig, DatasetConfig, TrainingArgsConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments: ", model_args)
    print("Data Arguments: ", data_args)
    print("Training Arguments: ", training_args)

    dir = training_args.output_dir.split('/')[-1]
    training_args.run_name = f'{dir}_{training_args.num_train_epochs}ep_{training_args.per_device_train_batch_size}b_{training_args.per_device_eval_batch_size}b'
    transformers.set_seed(training_args.seed)
    model_path = model_args.model_path
    # random_seed = 42
    train_path =  data_args.train_file # 'subtaskA_train_multilingual.jsonl'
    dev_path =  data_args.dev_file # 'subtaskA_dev_multilingual.jsonl'
    # test_path =  data_args.test_file # 'subtaskA_multilingual.jsonl'
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    prediction_path = training_args.output_dir+'/predictions' # For example subtaskB_predictions.jsonl
    
    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(dev_path):
       logging.error("File doesnt exists: {}".format(dev_path))
       raise ValueError("File doesnt exists: {}".format(dev_path))
    
    # if not os.path.exists(test_path):
    #     logging.error("File doesnt exists: {}".format(test_path))
    #     raise ValueError("File doesnt exists: {}".format(test_path))
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, truncate=True, model_max_length=400)

    def tokenizer_func(text):
        return tokenizer.tokenize(text)
    
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, max_features=500, stop_words=stop_words_list)
    train_df, valid_df, test_df = get_data(train_path, dev_path, vectorizer)#, dev_path, test_path, vectorizer)
    

    print('Loaded data.')
    print('................Starting preprocessing................')

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id#, ignore_mismatched_sizes = True
    )

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    temp = train_dataset.map(preprocess_function_top, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    tokenized_train_dataset = tokenized_train_dataset.add_column('input_ids_2', temp['input_ids'])
    tokenized_train_dataset = tokenized_train_dataset.add_column('attention_mask_2', temp['attention_mask'])

    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    temp = valid_dataset.map(preprocess_function_top, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = tokenized_valid_dataset.add_column('input_ids_2', temp['input_ids'])
    tokenized_valid_dataset = tokenized_valid_dataset.add_column('attention_mask_2', temp['attention_mask'])
    
    del temp, valid_dataset, train_dataset, vectorizer, train_df, valid_df

    # tokenize data for train/valid
    # tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    # tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    
    print('Train:', tokenized_train_dataset[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logger.info('\n................Start training................')
        # logger.info("Training...")
        logger.info("*** Train Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_train_dataset)}")
        logger.info("*** Dev Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_valid_dataset)}")

        trainer.train()

        logger.info("Training completed!")

        # save best model
        best_model_path = training_args.output_dir+'/best/'
        
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        
        trainer.save_model(best_model_path)


    if training_args.do_train:
        is_best=False
    else:
        is_best=True

    if training_args.do_predict:
        logger.info("\n................Start predicting................")

        if is_best:
            tokenizer = AutoTokenizer.from_pretrained(
                '/home/anastasiia.demidova/InfR/project/experiments/'+dir+'/best'
                )

            # load best model
            model = AutoModelForSequenceClassification.from_pretrained(
                '/home/anastasiia.demidova/InfR/project/experiments/'+dir+'/best',
                num_labels=len(label2id), id2label=id2label, label2id=label2id#, local_files_only=True
            )
        
        test_dataset = Dataset.from_pandas(test_df)
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})

        if is_best:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

        logger.info(f"*** Test Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_test_dataset)}")

        # predictions, _, _ = trainer.predict(tokenized_test_dataset)
        predictions = trainer.predict(tokenized_test_dataset)
        
        # prob_pred = softmax(predictions.predictions, axis=-1) #???????????????????
        preds = np.argmax(predictions.predictions, axis=-1)
        metric = evaluate.load("bstrai/classification_report")
        results = metric.compute(predictions=preds, references=predictions.label_ids)
        
        logger.info("Predictions completed!")
        logging.info(results)

        predictions_df = pd.DataFrame({'id': test_df['id'], 'label': preds})
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        predictions_df.to_json(prediction_path+'/subtask_a_monolingual.jsonl', lines=True, orient='records')



