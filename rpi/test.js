// Load the http module to create an http server.
var http = require('http');
// var fs = require('fs');
// var temp_path = '/sys/class/thermal/thermal_zone0/temp'
var py_path = 'sensors.py';
var PythonShell = require('python-shell');
var array
var py = new PythonShell(py_path);

py.on('message', function(message) {
	array = message.split(' ');
});
console.log(array)
