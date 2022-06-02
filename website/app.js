var express = require('express'); // require express module to run frontend server 

var app = express(); // declarations

app.use(express.static('.')); // can access and use html pages in the current directory


app.get('/', function(req, res) {
    res.render('index.html'); // when we enter localhost:8000 in browser, we tell the app to render/display index.html
})


// boiler plate code for building a POST http request
async function postData(url = '', data = {}) {
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      mode: 'cors', // no-cors, *cors, same-origin
      cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      credentials: 'same-origin', // include, *same-origin, omit
      headers: {
        'Content-Type': 'application/json'
      },
      redirect: 'follow', // manual, *follow, error
      referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
}

// this function gets called when user clicks 'submit' button in index.html
// argument to this function is the value user enters in the input text field
function sendReq(req) {
    console.log('req = ' + req) // log the request value

    var response_from_server;

    // call the postData function with url = backend server url
    // and request body = json object with inputText = user entered data
    postData("http://localhost:5005/input", { "inputText": req })
        .then(data => { // we process the response received from server
            console.log(data); // JSON data parsed by `data.json()` call
            if(data['success'] == true) { // set response accordingly
              response_from_server = data['answer']
            } else {
              response_from_server = data['message']
            }
            console.log(response_from_server) // log response
            // set the read only response field with data received from server
            document.getElementById("AnswerText").textContent = response_from_server; 
    });
      
}

// VERY IMP. Tell the app which port to run on. Should NOT be the same as the backend server port
app.listen(8000);