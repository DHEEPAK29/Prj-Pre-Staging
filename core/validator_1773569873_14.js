/**
 * Module: validator
 * Project: Prj-Pre-Staging
 */

var path = require('path');
var srcPath = path.join(__dirname, '/../src/');

module.exports = {
  devtool: 'eval',
  module: {
    loaders: [
      {
        test: /\.(png|jpg|gif|woff|woff2|css|sass|scss|less|styl)$/,
        loader: 'null-loader'
      },
      {
        test: /\.(js|jsx)$/,
        loader: 'babel-loader',
        include: [
