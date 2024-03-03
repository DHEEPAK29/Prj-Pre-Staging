/* Logic sourced from github.com/dockersamples/docker-swarm-visualizer */
var webpackCfg = require('./webpack.config');

module.exports = function(config) {
  config.set({
    basePath: '',
    browsers: ['PhantomJS'],
    files: [
      'test/loadtests.js'
    ],
    port: 8080,
    captureTimeout: 60000,
    frameworks: ['phantomjs-shim', 'mocha', 'chai'],
    client: {
      mocha: {}
    },
