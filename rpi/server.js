// Load the http module to create an http server.
var http = require('http');


var py_path = 'sensors.py';
var PythonShell = require('python-shell'); 


// Configure our HTTP server to respond with Hello World to all requests.
var server = http.createServer(function (request, response) {
     response.writeHead(200, {"Content-Type": "application/json"});
	 
	 var py = new PythonShell(py_path);
	 py.on('message', function(message) {
    
	 array = message.split(' ') 
	 
	 var temp = array[0];
	 var press = array[1];
	 var humid = array[2];

     var now = new Date();
     now = now.toJSON();

     var json = JSON.stringify({'temp': temp,'pressure': press,'humidity':humid,'time': now});
     response.end(json);
	 });
});
 
// Listen on port 8000, IP defaults to 127.0.0.1
server.listen(8000);
 
// Put a friendly message on the terminal
console.log("Server running at http://127.0.0.1:8000/");
