import flask
from flask import request, jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app) # This is required for this backend server running on port 5000 
          # to accept and process requests from another server
          # (in our case, frontend express javascript server running on port 8000)
          # running on a different port. ie., allow cross-origin requests.

# The path where the backend server is listening at
# This resolves to http://localhost:5005/input
@app.route("/input", methods=["POST"])
def input():
    body = request.json # read the request body
    print(body)

    data = {"success": False}

    if "inputText" not in body: # validate request body
        data["message"] = "Must include inputText in request body"
        return jsonify(data)

    input = 1    
    if "inputText" in body: # read user entered data from request body
        input = body["inputText"]
    
    try:
        output = int(input) * 3.1414 # you can replace this line with your model's output
        data["answer"] = output # build response 
        data["success"] = True
    except:
        data["message"] = "Must input a sentence to run the model."

    return jsonify(data) # return response 

# This is the port where this backend server port runs on
if __name__ == '__main__':
    app.run(host= "0.0.0.0", port=5005)