import os
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from vit import ViT, PerformerViT
import io

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # 部署生产环境时请更改此密钥

# TODO: 根据你训练的模型架构参数进行修改
# ViT模型参数（示例）
vit_params = {
    'image_size': 32,
    'patch_size': 4,
    'num_classes': 10,
    'dim': 200,
    'depth': 6,
    'heads': 8,
    'mlp_dim': 256,
    'pool': 'cls',
    'channels': 3,
    'dim_head': 64,
    'dropout': 0.15,
    'emb_dropout': 0.15,
    'qkv_bias': True
}

# Performer模型参数（示例）
performer_params = {
    'image_size': 32,
    'patch_size': 4,
    'num_classes': 10,
    'dim': 128,
    'depth': 6,
    'heads': 8,
    'mlp_dim': 200,
    'pool': 'cls',
    'channels': 3,
    'dim_head': 32,
    'dropout': 0.15,
    'emb_dropout': 0.15,
    'qkv_bias': True,
    'nb_features': 64,
    'generalized_attention': True,
    'kernel_fn': nn.ReLU(),
    'no_projection': False
}

VIT_MODEL_PATH = os.path.join('model', 'vit_model32.pth')
PERFORMER_MODEL_PATH = os.path.join('model', 'performer_relu_model64.pth')

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

models = {}
infer_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
def load_model(model_type):
    if model_type == 'vit':
        if 'vit' not in models:
            model = ViT(**vit_params)
            state_dict = torch.load(VIT_MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            models['vit'] = model
        return models['vit']
    elif model_type == 'performer':
        if 'performer' not in models:
            model = PerformerViT(**performer_params)
            state_dict = torch.load(PERFORMER_MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            models['performer'] = model
        return models['performer']
    else:
        raise ValueError("Unknown model type")

def preprocess_image(img):
    # img is PIL Image
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    img = img.crop((left, top, right, bottom))

    x = infer_transform(img)
    x = x.unsqueeze(0)  # [1,3,32,32]
    return x

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No image file uploaded")
            return redirect(url_for('index'))

        file = request.files['image']
        if file.filename == '':
            flash("No selected file")
            return redirect(url_for('index'))

        model_type = request.form.get('model_type', None)
        if model_type not in ['vit', 'performer']:
            flash("No model selected or invalid model type")
            return redirect(url_for('index'))

        model = load_model(model_type)

        img = Image.open(file.stream).convert('RGB')
        x = preprocess_image(img)

        with torch.no_grad():
            preds = model(x)
            pred_class = torch.argmax(preds, dim=1).item()
            class_name = CLASSES[pred_class]

        # 返回结果页面或者JSON都可以，这里返回到一个结果页
        return render_template('result.html', label=class_name, model_type=model_type)

    else:
        # GET请求，返回上传页面
        return render_template('upload.html')

if __name__ == '__main__':
    # 本地测试
    app.run(host='0.0.0.0', port=8080, debug=True)
