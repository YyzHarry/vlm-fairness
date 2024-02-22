import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils import data
from tqdm import tqdm

import models.clip as clip
from models.model import CLIP
from dataset import datasets
import config
from utils import misc
from utils.eval import evaluate, sigmoid


def load_clip(model_path, pretrained=False, context_length=77):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained is False:
        # use new model params
        params = {
            'embed_dim': 768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length,
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }
        model = CLIP(**params)
    else:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model


def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    This function outputs the weights for each of the classes based on the 
    output of the trained clip model text transformer. 

    args: 
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis', ...]).
    * templates - Python list of phrases that will be independently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.

    Returns PyTorch Tensor, output of the text encoder given templates.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname)[6:] if template.format(classname).startswith("no No ")
                     else template.format(classname) for template in templates]
            texts = clip.tokenize(texts, context_length=context_length)  # tokenize
            class_embeddings = model.encode_text(texts.to(device))  # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates
            class_embedding = class_embeddings.mean(dim=0)
            # norm over new averaged templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=False):
    """
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images and the text embeddings.

    args:
        * loader - PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.

    Returns numpy array, predictions on all test data samples.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_pred = []
    with torch.no_grad():
        for _, x in tqdm(loader):
            x = x.to(device)
            # predict
            image_features = model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # (bsz, 768)

            logits = image_features @ zeroshot_weights  # (bsz, num_classes)
            logits = logits.data.cpu().numpy()

            if softmax_eval is False:
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits)

            y_pred.append(logits)

            if verbose:
                plt.imshow(x[0][0])
                plt.show()
                print('images: ', x)
                print('images size: ', x.size())
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())

    y_pred = np.vstack(y_pred)
    return y_pred


def make(model_path, dataset, data_path, data_split, pretrained=True, context_length=77):
    # load model
    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,
        context_length=context_length
    )
    # create dataset
    torch_dset = vars(datasets)[dataset](data_path, data_split)
    loader = torch.utils.data.DataLoader(
        dataset=torch_dset,
        batch_size=1024,
        num_workers=8,
        shuffle=False
    )
    return model, loader


def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77):
    """
    This function will make probability predictions for a single template (i.e. "has {}").
    
    args:
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis', ...])
        * template - string, template to input into model
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation
        * context_length (optional) - int, max number of tokens of text inputted into the model

    Returns list, predictions from the given template.
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred


def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77):
    """ Run softmax evaluation to obtain a single prediction from the model. """
    # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(eval_labels, pos, model, loader,
                                     softmax_eval=True, context_length=context_length)
    neg_pred = run_single_prediction(eval_labels, neg, model, loader,
                                     softmax_eval=True, context_length=context_length)

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred


def ensemble_models(model_paths, cxr_labels, cxr_pair_template, cache_dir, dataset, data_path, data_split):
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged set of predictions.
    """
    predictions = []
    model_paths = sorted(model_paths)

    avg_pred_path = Path(cache_dir) / "avg_pred.npy"
    if os.path.exists(avg_pred_path):
        print(f"Loading cached ensemble predictions...")
        y_pred_avg = np.load(avg_pred_path)
        return y_pred_avg

    for path in model_paths:
        model_name = Path(path).stem

        model, loader = make(
            model_path=path,
            dataset=dataset,
            data_path=data_path,
            data_split=data_split)

        cache_path = Path(cache_dir) / f"{model_name}.npy"
        if os.path.exists(cache_path):
            print(f"Loading cached prediction for [{model_name}]...")
            y_pred = np.load(cache_path)
        else:
            print(f"Inferring model [{path}]...")
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None:
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)

    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    np.save(file=avg_pred_path, arr=y_pred_avg)
    return y_pred_avg


def run_ensemble_zero_shot(cxr_labels, cxr_templates, model_dir, cache_dir, data_path, dataset, data_split):
    np.random.seed(97)
    models = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            full_dir = os.path.join(subdir, file)
            models.append(full_dir)

    avg_pred = ensemble_models(
        model_paths=models,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_templates,
        cache_dir=cache_dir,
        dataset=dataset,
        data_path=data_path,
        data_split=data_split
    )

    test_labels = []
    test_preds = []
    for task in config.TASKS[dataset]:
        df = pd.read_csv(config.METADATA[dataset])
        df = df[df['split'] == config.SPLITS[data_split]]
        test_labels.append(df[task].astype(int).tolist())
        test_preds.append(avg_pred[:, cxr_labels.index(task)])

    test_labels = np.transpose(np.vstack(test_labels), (1, 0))
    test_preds = np.transpose(np.vstack(test_preds), (1, 0))
    evaluate(test_preds, test_labels, config.TASKS[dataset], verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default="MIMIC", choices=datasets.DATASETS)
    parser.add_argument('--split', type=str, default="te", choices=['tr', 'va', 'te'])
    # prompt template
    parser.add_argument('--template', type=str, default="pathology", help="JSON filename for prompt template")
    # others
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--predictions_dir', type=str, default='predictions')
    args = parser.parse_args()

    with open(f'configs/template/{args.template}.yaml', 'r') as file:
        prompt_templates = yaml.safe_load(file)
    prompt_templates = [eval(v) for _, v in prompt_templates.items()][0]
    assert isinstance(prompt_templates, tuple)

    output_dir = Path(args.predictions_dir) / args.template / f"{args.dataset}_{args.split}"
    output_dir.mkdir(exist_ok=True, parents=True)
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))

    run_ensemble_zero_shot(
        cxr_labels=config.TASKS[args.dataset],
        cxr_templates=prompt_templates,
        model_dir=args.model_dir,
        cache_dir=output_dir,
        data_path=args.data_dir,
        dataset=args.dataset,
        data_split=args.split
    )
