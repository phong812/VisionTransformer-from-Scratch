from flask import Flask, request, render_template, jsonify
import torch
import json
from PIL import Image
import model
import io
from utils import create_vit_pretrained_model


app = Flask(__name__)
food101_image_class = json.load(open('food101_image_class.json'))
model_pre, transform_pre  = create_vit_pretrained_model(num_classes=10, seed=42)
model_pre.load_state_dict(torch.load('models/vit_pre_1.pt'))
model_pre.eval()

def get_prediction(image_bytes):
    tensor = transform_pre(Image.open(io.BytesIO(image_bytes))).unsqueeze(0)
    outputs = model_pre.forward(tensor)
    pred = torch.nn.functional.softmax(outputs, dim=1)
    _, prob = torch.max(pred, dim=1)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    
    return pred, food101_image_class[predicted_idx]    

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['imagefile']
        img_bytes = file.read()
        prob, class_name = get_prediction(img_bytes)
        prob_class = max(max(prob))
        a = {'class_name': class_name, 'prob_x': prob_class.tolist()}
        return render_template('index.html', prediction=a)
        # return jsonify({'class_name': class_name, 'prob_x': prob_class.tolist()})

# @app.route('/', methods=['GET'])
# def hello_world():
#     return render_template('index.html')


# @app.route('/predict', method=['POST'])
# def predict():
#     image_file = request.files['imagefile']
#     image_path = './images/' + image_file.filename
#     image_file.save(image_path)
    
#     img = Image.open(image_path).convert("RGB")
#     img_transform = transform_pre(img).unsqueeze(0)
#     pred = torch.softmax(model_pre(img_transform), dim=1)
    
#     return render_template('index.html', prediction=)    

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)