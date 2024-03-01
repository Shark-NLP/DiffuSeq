from diffuseq.transformer_model import TransformerNetModel
from transformers import BertTokenizer

import torch
import torch.nn as nn
import torch.optim as optim

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
input_dims = 128
output_dims = 128
hidden_t_dim = 128

model = TransformerNetModel(
    input_dims=input_dims,
    output_dims=output_dims,
    hidden_t_dim=hidden_t_dim,
    vocab_size=vocab_size,
    dropout=0.1
)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 80 

dataloader = 

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:  
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



model.eval()  
with torch.no_grad():
    predictions = model(input_data)  



torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))
model.eval()
