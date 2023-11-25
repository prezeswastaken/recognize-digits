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


@app.route("/")
def index():
    data, target = random.choice(test_data)
    data = data.unsqueeze(0).to(device)
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()

    image = data.squeeze(0).squeeze(0).cpu().numpy()

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray((image * 255).astype("uint8"))

    # Convert the PIL Image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template(
        "index.html", prediction=f"I guess this is {prediction}", image=image_base64
    )


if __name__ == "__main__":
    app.run(debug=True)
