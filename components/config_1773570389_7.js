/**
 * Module: config
 * Project: Prj-Pre-Staging
 */

var path = require('path');

var port = 8080;
var srcPath = path.join(__dirname, '/../src');
var publicPath = '/';

module.exports = {
  port: port,
  debug: true,
  output: {
    path: path.join(__dirname, '/../dist'),
    filename: 'app.js',
    publicPath: publicPath
  },
  devServer: {
