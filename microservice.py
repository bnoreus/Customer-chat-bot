# This Python file uses the following encoding: utf-8
import os

import json
from bottle import route, run, debug, template, request, response, static_file, error, hook, BaseRequest, Bottle
import requests
import random
from time import time


BaseRequest.MEMFILE_MAX = 10000000000 # (or whatever you want)

app = Bottle()


# Add JSON endpoint headers
@app.hook('before_request')
def enable_cors():
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers["Content-type"] = "application/json"

@app.route("/")
def index():
	response.headers["Content-type"] = "text/html"
	return template("html_pages/index.html")

@app.route("/html/<filepath:re:.*\.html>")
def html(filepath):
	return static_file(filepath,root="html_pages/")

@app.route("/css/<filepath:re:.*\.css>")
def css(filepath):
	return static_file(filepath,root="css/")

@app.route("/js/<filepath:re:.*\.js>")
def js(filepath):
	return static_file(filepath,root="js/")

debug(True)
app.run(host='0.0.0.0',port=1200)
