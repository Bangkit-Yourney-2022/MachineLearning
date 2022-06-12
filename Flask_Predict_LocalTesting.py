# Import Library
from flask import Flask,request,jsonify
from tensorflow import keras 
import numpy as np
import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load Data
model = keras.models.load_model('capstone_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as a:
    lb = pickle.load(a)

stopwords = []
with open('stopwords.txt', encoding='utf-8') as f:
  for line in f:
    stopwords.append(line.strip())

# Function
def clean(a):
    temp = [w for w in a.split() if w not in stopwords]
    temp = " ".join(temp)
    temp = re.sub(r'[.,’"\'-?:!;]', '', temp)
    temp = re.sub(r'^whats|^im', '', temp)
    temp = temp.strip()
    return temp

# Request
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    user_input = request.form.get('user')
    user_input = clean(str(user_input))
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                             truncating='post', maxlen=100))[0]

    return jsonify({'Answer':str(lb.inverse_transform([np.argmax(result)]))})
if __name__ == '__main__':
    app.run(debug=True)