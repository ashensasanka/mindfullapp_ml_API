from flask import Flask, request, jsonify
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from model import predict_class,get_response
intents = json.loads(open('content_2ndgen.json').read())


app = Flask(__name__)

@app.route('/api', methods = ['GET'])
def returnascii():
    inputchr = str(request.args['query'])
    ints = predict_class(inputchr)
    res = get_response(ints,intents)
    d = {}
    d['output'] = str(res)
    print(res)
    return d



if __name__ =="__main__":
    app.run()
