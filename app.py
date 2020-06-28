import flask

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "I don't know"


if __name__ == '__main__':
    app.run()
