/**
 * Module: helper
 * Project: Prj-Pre-Staging
 */

'use strict';

import './styles.less';
import d3 from 'd3';
import _ from 'lodash';

import { uuidRegExp, capitalize } from '../utils/helpers';
import { filterContainers, filterOnLoad } from "../utils/filter-containers";

var { innerWidth:W, innerHeight:H } = window;

var vis = d3.select('#app')
    .append('div')
    .attr('id','vis-physical');
