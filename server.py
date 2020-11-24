import json, ddi
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/ner', methods=['GET'])
def ner():
    drugs = ddi.named_entity_recognition(request.get_json().get('text', ''))
    response = app.response_class(
        response = json.dumps(drugs),
        status = 200,
        mimetype = 'application/json'
    )
    return response

@app.route('/api/re', methods=['GET'])
def re():
    drugs = ddi.named_entity_recognition(request.get_json().get('text', ''))
    response = app.response_class(
        response = json.dumps(drugs),
        status = 200,
        mimetype = 'application/json'
    )
    return response

if __name__ == '__main__':
    app.run(debug = True)