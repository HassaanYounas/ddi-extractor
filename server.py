import json, ddi
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/input/text')
def input_text():
    return render_template('/input/ddi-text.html')

@app.route('/input/file')
def input_file():
    return render_template('/input/ddi-file.html')

@app.errorhandler(404)
def error(e):
    return render_template('404.html'), 404

@app.route('/api/ner', methods = ['POST'])
def ner():
    drugs = ddi.named_entity_recognition(request.get_json().get('text', ''))
    response = app.response_class(
        response = json.dumps(drugs),
        status = 200,
        mimetype = 'application/json'
    )
    return response

@app.route('/api/re', methods = ['POST'])
def re():
    ddis = ddi.relation_extraction(request.get_json().get('text', ''))
    response = app.response_class(
        response = json.dumps(ddis),
        status = 200,
        mimetype = 'application/json'
    )
    return response

if __name__ == '__main__':
    app.run(debug = True)