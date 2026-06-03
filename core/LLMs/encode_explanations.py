import torch
from transformers import AutoTokenizer, AutoModel


def encode_explanations(explanations):
    model_name = "microsoft/deberta-base"
    print("loading encoder for explanations...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        inputs = tokenizer(
            explanations,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].to("cpu")

    return cls_emb


def encode_explanations_batch(explanations, batch_size=128):
    model_name = "microsoft/deberta-base"
    print("loading encoder for explanations...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cls_embs = []
    with torch.no_grad():
        for i in range(0, len(explanations), batch_size):
            batch_explanations = explanations[i : i + batch_size]
            inputs = tokenizer(
                batch_explanations,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].to("cpu")
            cls_embs.append(cls_emb)

    return torch.cat(cls_embs, dim=0)
