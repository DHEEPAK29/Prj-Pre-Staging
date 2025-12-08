/**
 * Module: handler
 * Project: Prj-Pre-Staging
 */

var path = require('path');
var webpack = require('webpack');
var _ = require('lodash');

var baseConfig = require('./base');

var config = _.merge({
  entry: [
    'webpack-dev-server/client?http://127.0.0.1:8080',
    'webpack/hot/only-dev-server',
    './src/main'
  ],
  cache: true,
  devtool: 'eval',
  plugins: [
