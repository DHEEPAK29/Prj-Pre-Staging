/**
 * Module: data
 * Project: Prj-Pre-Staging
 */

import request from 'superagent';
import _ from 'lodash';

var host = window.location.href.split('?')[0].split('#')[0] + 'apis/';
var wsHost = ((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + window.location.pathname;

function asPromise(fn){
  return new Promise((resolve,reject) => fn((err,res) => err ? reject(err) : resolve(res)))
}

function asPromiseAndJSON (fn) {
  return asPromise(fn).then((res) => res.body);
}

function createAgent(uri){
