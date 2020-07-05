from flask import Flask,request
from gevent.pywsgi import WSGIServer
import json

import base64
from io import StringIO,BytesIO 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
import os
import random
import numpy as np
from collections import OrderedDict
import string
from PIL import Image

app=Flask(__name__)

@app.route("/sina",methods=['POST'])
def captcha_predict():
    return_dict= {'return_code': '200', 'return_info': '处理成功', 'result': False}
    get_data= request.form.to_dict()
    # print(request.form)
    if 'img' in get_data.keys():
        base64_img = request.form['img']
        try:
            return_dict['result'] = predict(base64_img=base64_img)
        except Exception as e:
            return_dict['result'] = str(e)
            return_dict['return_info'] = '模型识别错误,确认图片大小。并确认base64编码'
    else:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '参数错误，没有img属性'
    return json.dumps(return_dict, ensure_ascii=False)

@app.route("/sina",methods=['GET'])
def get_info():
    return_dict= { 'return_info': '请用post方法请求该地址'}
    return json.dumps(return_dict)


# 基本的参数
characters = '-' + string.digits + string.ascii_lowercase
width, height, n_len, n_classes = 100, 40, 5, len(characters)
n_input_length = 12
# print(characters, width, height, n_len, n_classes)


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        # pools = [2, 2, 2, 2, (2, 1)]
        # 减少一个池化层
        pools = [2, 2, 2, (2, 1)]
        modules = OrderedDict()
        
        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)
        
        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        
        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)
    
    def infer_features(self):
        x = torch.zeros((1,)+self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


""" 加载训练后的模型 """
model = Model(n_classes, input_shape=(3, height, width))
model.load_state_dict(torch.load('ctc_625_22.pth',map_location=torch.device('cpu')))
model.eval()

def predict(base64_img):
    image = Image.open(BytesIO(base64.b64decode(base64_img)))
    # 转换成3通道
    image = to_tensor(image.convert("RGB"))
    output = model(image.unsqueeze(0))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    pred_str = decode(output_argmax[0])
    # print('pred:', pred_str)
    return pred_str


if __name__ == "__main__":
    # app.run(debug=True)
    print('启动服务')
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
    