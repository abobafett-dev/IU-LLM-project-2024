import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class BERTClass(torch.nn.Module):
    def __init__(self, num_classes):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear1 = torch.nn.Linear(1024, 2048)
        self.linear2 = torch.nn.Linear(2048, num_classes)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear1(output_dropout)
        output = F.leaky_relu(self.linear2(output))
        return output


with open('tags2.txt', 'r') as f:
    data_tags = f.read().lower().split('\n')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BERTClass(len(data_tags))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("output\\MLTC_model_state_new_loss.bin"))
model.to(device)


def tags(text: str) -> list[str]:
    global model
    global tokenizer
    global device

    encoded_text = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    output = model(input_ids, attention_mask, token_type_ids)
    output = output.detach().cpu()
    _, idx = torch.topk(output.flatten(), k=5, dim=0)
    list_my = [data_tags[i] for i in idx]
    return list_my

#%%
