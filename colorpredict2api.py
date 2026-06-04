from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/get-route')
def get_route():
    route = [
        {"latitude": 19.0760, "longitude": 72.8777},
        {"latitude": 19.0785, "longitude": 72.8800},
        {"latitude": 19.0810, "longitude": 72.8850}
    ]
    return jsonify({"route": route})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)