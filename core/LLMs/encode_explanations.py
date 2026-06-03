import torch
from transformers import AutoModel, AutoTokenizer

_ENCODER = "microsoft/deberta-base"


def encode_explanations(explanations):
    """Encode LLM explanation strings with a frozen PLM (CLS token)."""
    print("loading encoder for explanations...")
    tokenizer = AutoTokenizer.from_pretrained(_ENCODER)
    model = AutoModel.from_pretrained(_ENCODER)
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
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
    return cls_emb
