import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from transformers import BertModel, BertConfig
import numpy as np
from pytesseract import pytesseract
import argparse

# 1. Input Processing: OCR and Tokenization
def extract_ocr_data(document_path):
    ocr_results = pytesseract.image_to_data(document_path, output_type=pytesseract.Output.DICT)
    tokens = ocr_results['text']
    coordinates = [(ocr_results['left'][i], ocr_results['top'][i]) 
                   for i in range(len(tokens)) if ocr_results['text'][i].strip()]
    tokens = [t for t in tokens if t.strip()]
    return tokens, coordinates

# 2. GCN for Super-Tokens
class GCNSuperToken(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GCNSuperToken, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = self.relu(layer(g, h))
        return h

def build_graph(coordinates, tokens, threshold=50):
    g = dgl.DGLGraph()
    g.add_nodes(len(tokens))
    edge_list = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = np.sqrt((coordinates[i][0] - coordinates[j][0])**2 + 
                           (coordinates[i][1] - coordinates[j][1])**2)
            if dist < threshold:
                edge_list.append((i, j))
                edge_list.append((j, i))
    g.add_edges(*zip(*edge_list))
    return g

# 3. Rich Attention Mechanism
class RichAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RichAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.spatial_weight = nn.Parameter(torch.randn(1))

    def forward(self, queries, keys, values, coordinates):
        q = self.query(queries)
        k = self.key(keys)
        v = self.value(values)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        spatial_dists = torch.cdist(coordinates, coordinates)
        spatial_penalty = self.spatial_weight * spatial_dists
        attention_scores = attention_scores - spatial_penalty
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

# 4. FormNet Model
class FormNet(nn.Module):
    def __init__(self, bert_config, gcn_input_dim, gcn_hidden_dim, num_labels):
        super(FormNet, self).__init__()
        self.bert = BertModel(bert_config)  # Replaced ETC with BERT
        self.gcn = GCNSuperToken(gcn_input_dim, gcn_hidden_dim)
        self.rich_attention = RichAttention(gcn_hidden_dim)
        self.classifier = nn.Linear(gcn_hidden_dim, num_labels)

    def forward(self, graph, token_features, coordinates, input_ids, attention_mask):
        super_tokens = self.gcn(graph, token_features)
        rich_output, _ = self.rich_attention(super_tokens, super_tokens, super_tokens, coordinates)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        combined = rich_output + bert_output[:, :rich_output.size(1), :]
        logits = self.classifier(combined)
        return logits

# 5. Training Loop
def train_formnet(model, graph, token_features, coordinates, input_ids, attention_mask, labels, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(graph, token_features, coordinates, input_ids, attention_mask)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# 6. Main Function
def main():
    parser = argparse.ArgumentParser(description="FormNet Document Extraction")
    parser.add_argument("--document", type=str, default="sample_form.pdf", help="Path to input document")
    args = parser.parse_args()

    # Extract tokens and coordinates
    document_path = args.document
    tokens, coordinates = extract_ocr_data(document_path)

    # Dummy data for testing
    token_features = torch.randn(len(tokens), 768)
    coordinates = torch.tensor(coordinates, dtype=torch.float)
    input_ids = torch.randint(0, 1000, (1, len(tokens)))
    attention_mask = torch.ones(1, len(tokens))
    labels = torch.randint(0, 10, (1, len(tokens)))

    # Build graph
    graph = build_graph(coordinates, tokens)

    # Initialize model
    bert_config = BertConfig(max_position_embeddings=len(tokens))
    model = FormNet(bert_config, gcn_input_dim=768, gcn_hidden_dim=768, num_labels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # Train
    loss = train_formnet(model, graph, token_features, coordinates, input_ids, attention_mask, labels, optimizer)
    print(f"Training loss: {loss}")

if __name__ == "__main__":
    main()