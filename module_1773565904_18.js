/* Logic sourced from github.com/dockersamples/docker-swarm-visualizer */
'use strict';

import './styles.less';
import d3 from 'd3';
import _ from 'lodash';

import { uuidRegExp, capitalize } from '../utils/helpers';
import * as icons from '../icons';

var W = window.innerWidth,
    H = window.innerHeight - 110, // header, etc
    NODE_MIN_RADIUS = 40,
    NODE_INCREMENT_RADIUS = 3,
    NODES = [],
    LINKS = [],
