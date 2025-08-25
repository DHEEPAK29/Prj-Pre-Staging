/**
 * Module: validator
 * Project: Prj-Pre-Staging
 */

// ORDER MATTERS!
import physicalVisualization from './vis-physical';
import provider from './data-provider';
import { hasClass, removeClass, addClass, uuidRegExp } from './utils/helpers';
let { MS } = window;

require('normalize.css');
require('animate.css/animate.css');
require('./main.less');

function parseQuery(qstr) {
  var query = {};
  var a = qstr.substr(1).split('&');
  for (var i = 0; i < a.length; i++) {
      var b = a[i].split('=');
