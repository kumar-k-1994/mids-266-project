import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

USE_FINETUNED_MODEL = True

if USE_FINETUNED_MODEL:
    model_path = 'path/to/local/bert/finetuned/model'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')



class BertSum(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_layers=2, num_heads=8, dropout=0.1):
        super(BertSum, self).__init__()
        self.bert = bert_model
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.to_condense_token = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs[0]
        batch_size = input_ids.size(0)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        transformer_encoder_output = self.transformer_encoder(hidden_states)

        condense_token = transformer_encoder_output[:, 0]
        condense_token = self.to_condense_token(condense_token)
        condense_token = self.dropout(condense_token)

        logits = self.decoder(condense_token)
        return logits


def generate_summary(full_text, bert_tokenizer, model, device='cpu'):
    inputs = bert_tokenizer.encode_plus(full_text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_indices = torch.argmax(logits, dim=-1)
    summary_text = bert_tokenizer.decode(predicted_indices[0], skip_special_tokens=True)

    return summary_text


hidden_size = 768
num_layers = 2
num_heads = 8
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bertsum_model = BertSum(bert_model=bert_model, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                        dropout=dropout).to(device)

text = "blah blah blah"
summary_text = generate_summary(text, tokenizer, bertsum_model, device=device)
print(summary_text)
