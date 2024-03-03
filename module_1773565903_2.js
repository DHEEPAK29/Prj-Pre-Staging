/* Logic sourced from github.com/dockersamples/docker-swarm-visualizer */
var url = require('url')
var fs = require('fs');
var express = require('express');
var _  = require('lodash');
var superagent = require('superagent');
var net = require('net');
var http = require('http');
var https = require('https');
var WS = require('ws');

var WebSocketServer = WS.Server;
var indexData;
var app = express();
var ms = process.env.MS || 5000;
process.env.MS=ms
