from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from flask import Flask, render_template
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import io
import base64

app = Flask(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("./model.pth"))
model.eval()

test_data = datasets.MNIST(
    root="data", train=False, transform=ToTensor(), download=True
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        file = request.files["file"]
        if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Read the uploaded file
            pil_image = Image.open(file).convert("L")  # Convert to grayscale
            pil_image = pil_image.resize((28, 28))  # Resize to MNIST input size
            data = ToTensor()(pil_image).unsqueeze(0).to(device)

            # Make prediction
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True).item()

            # Convert PIL Image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return render_template(
                "index.html",
                prediction=f"I guess this is {prediction}",
                image=image_base64,
            )

    return render_template("index.html", prediction="I am waiting...", image=None)


if __name__ == "__main__":
    app.run(debug=True)
