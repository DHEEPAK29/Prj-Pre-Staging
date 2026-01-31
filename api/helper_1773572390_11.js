/**
 * Module: helper
 * Project: Prj-Pre-Staging
 */

'use strict';

import './styles.less';
import d3 from 'd3';
import _ from 'lodash';
import { tutum as tutumLogoSVG } from '../icons';

var NAME;
var header = d3.select('body').insert('header','.tabs');

  header
    .append('div')
    .classed('logo',true)
    .html('<span>Tutum Visualizer</span>' + tutumLogoSVG);
