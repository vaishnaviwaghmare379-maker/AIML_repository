# Assignment 10: Fake News Detection from Multimodal Data (Text + Image)
# Install first:  pip install torch torchvision transformers pandas scikit-learn pillow
# Dataset format (CSV): text, image_path, label     (label: 1 = fake, 0 = real)
# Example public datasets: Fakeddit, MediaEval Verifying Multimedia Use, Twitter/Weibo fake-news sets
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR = 64, 16, 3, 2e-5

# ------------------------------------------------------------------ 1. Dataset
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
img_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class NewsDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        tok = tokenizer(row["text"], padding="max_length", truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        img = img_tf(Image.open(row["image_path"]).convert("RGB"))
        return (tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0),
                img, torch.tensor(row["label"], dtype=torch.long))

# ------------------------------------------------------------------ 2. Model
class MultimodalFakeNewsNet(nn.Module):
    """Text encoder (DistilBERT) + Image encoder (ResNet-18) -> late fusion -> classifier"""
    def __init__(self):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained(TEXT_MODEL)           # 768-d
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()                                       # 512-d
        self.img_enc = resnet
        self.text_proj = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 128),
                                        nn.ReLU(), nn.Linear(128, 2))
    def forward(self, ids, mask, img):
        t = self.text_enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        v = self.img_enc(img)
        fused = torch.cat([self.text_proj(t), self.img_proj(v)], dim=1)  # concatenation fusion
        return self.classifier(fused)

# ------------------------------------------------------------------ 3. Data loaders
df = pd.read_csv("fake_news_multimodal.csv")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_dl = DataLoader(NewsDataset(train_df), batch_size=BATCH, shuffle=True)
test_dl = DataLoader(NewsDataset(test_df), batch_size=BATCH)

model = MultimodalFakeNewsNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------------ 4. Training
for epoch in range(EPOCHS):
    model.train(); total = 0
    for ids, mask, img, y in train_dl:
        ids, mask, img, y = ids.to(DEVICE), mask.to(DEVICE), img.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask, img), y)
        loss.backward(); optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total / len(train_dl):.4f}")

# ------------------------------------------------------------------ 5. Evaluation
model.eval(); y_true, y_pred = [], []
with torch.no_grad():
    for ids, mask, img, y in test_dl:
        out = model(ids.to(DEVICE), mask.to(DEVICE), img.to(DEVICE))
        y_pred += out.argmax(1).cpu().tolist(); y_true += y.tolist()
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# ------------------------------------------------------------------ 6. Predict a new post
def predict(text, image_path):
    tok = tokenizer(text, padding="max_length", truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    img = img_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(tok["input_ids"], tok["attention_mask"], img), dim=1)[0]
    return {"Real": prob[0].item(), "Fake": prob[1].item()}

# print(predict("Shocking! Celebrity found alive on Mars", "sample.jpg"))
torch.save(model.state_dict(), "multimodal_fakenews.pt")
