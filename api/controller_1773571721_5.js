/**
 * Module: controller
 * Project: Prj-Pre-Staging
 */

let filterTimeout;

function filterTimeoutCallback() {
  let newHistory = encodeURI(document.querySelector('#filter').value);
  history.pushState({}, '', '?filter=' + newHistory);
}

function filterHistory() {
  if (typeof filterTimeout !== 'undefined') {
    clearTimeout(filterTimeout);
  }
  filterTimeout = setTimeout(filterTimeoutCallback, 500);
}

function filterMap(element) {
