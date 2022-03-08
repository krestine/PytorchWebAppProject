from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
# import os
import json
# import base64

# import torch
import numpy as np
# from torchvision import models, transforms
# from torch.autograd import Variable
# import torchvision.models as models

import onnx
import onnxruntime

from PIL import Image

import requests

# All the 1000 imagenet classes
class_labels = 'imagenet_classes.json'

# Read the json
with open('imagenet_classes.json', 'r') as fr:
    json_classes = json.loads(fr.read())

app = Flask(__name__)

# Allow
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

label_dict = {0: '사과', 1: '산', 2: '달', 3: '얼굴', 4: '문',
              5: '봉투', 6: '물고기', 7: '기타', 8: '별', 9: '번개'}

@app.route("/")
def hello():
    return "Hello World!"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("request data", request.data)
        print("request files", request.files)

        # check if the post request has the file part
        if 'file' not in request.files:
            # print("no file part")
            return "No file part"
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Send uploaded image for prediction
            # read_img = Image.open(UPLOAD_FOLDER + filename)
            read_img = Image.open(file)
            if filename.endswith('.png'):
                read_img = read_img.convert('RGB')
            predicted_image_class = predict_img(read_img)
            print("predicted_image_class", predicted_image_class)

    return json.dumps(predicted_image_class)


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')
    norm_img_data = img_data.reshape(1, 784).astype('float32')
    return norm_img_data

    # normalize
    ''' mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    # norm_img_data = norm_img_data.reshape(1, 1, 28, 28).astype('float32')
    norm_img_data = norm_img_data.reshape(1, 784).astype('float32')
    return norm_img_data'''


def predict_img(read_img):
    # Available model archtectures =
    # 'alexnet','densenet121', 'densenet169', 'densenet201', 'densenet161','resnet18',
    # 'resnet34', 'resnet50', 'resnet101', 'resnet152','inceptionv3','squeezenet1_0', 'squeezenet1_1',
    # 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19'

    # Choose which model achrictecture to use from list above
    # architecture = models.squeezenet1_0(pretrained=True)
    # architecture = onnx.load("squeezenet1.1-7.onnx")
    # onnx.checker.check_model(architecture)

    # ort_session = onnxruntime.InferenceSession("squeezenet1.1-7.onnx")
    ort_session = onnxruntime.InferenceSession("test_onnx.onnx")
    # ort_session = onnxruntime.InferenceSession("mobilenetv2-7.onnx")

    #def to_numpy(tensor):
        #return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # architecture.eval()

    # Normalization according to https://pytorch.org/docs/0.2.0/torchvision/transforms.html#torchvision.transforms.Normalize
    # Example seen at https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Preprocessing according to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Example seen at https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

    '''preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])'''
    # read_img = read_img.resize((224, 224))
    read_img = read_img.resize((28, 28))
    read_img = read_img.convert('L')

    # Path to uploaded image
    # path_img = img_path

    # Read uploaded image
    # read_img = Image.open(path_img)

    # Convert image to RGB if it is a .png
    # if path_img.endswith('.png'):
        # read_img = read_img.convert('RGB')

    # image_data = np.array(read_img).transpose(2, 0, 1)
    image_data = np.array(read_img)
    img_tensor = preprocess(image_data)
    # img_tensor.unsqueeze_(0)
    # img_variable = Variable(img_tensor)

    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    # Predict the image
    # outputs = architecture(img_variable)

    # print(ort_outs[0])
    # print(outputs)
    # np.testing.assert_allclose(to_numpy(outputs), ort_outs[0], rtol=1e-03, atol=1e-05)
    # Couple the ImageNet label to the predicted class
    labels = {int(key): value for (key, value)
              in json_classes.items()}
    # print("\n Answer: ", labels[outputs.data.numpy().argmax()])

    # return labels[outputs.data.numpy().argmax()]
    # return labels[ort_outs[0].argmax()]
    print(ort_outs)
    return label_dict[ort_outs[0].argmax()]


if __name__ == "__main__":
    app.run(debug=True)
