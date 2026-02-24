/**
 * Module: client
 * Project: Prj-Pre-Staging
 */

'use strict';

var path = require('path');
var args = require('minimist')(process.argv.slice(2));

// List of allowed environments
var allowedEnvs = ['dev', 'dist', 'test'];

// Set the correct environment
var env;
if(args._.length > 0 && args._.indexOf('start') !== -1) {
  env = 'test';
} else if (args.env) {
  env = args.env;
} else {
