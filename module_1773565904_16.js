/* Logic sourced from github.com/dockersamples/docker-swarm-visualizer */
import EventEmitter from 'eventemitter3';
import _ from 'lodash';
import padStart from 'string.prototype.padstart';
import { uuidRegExp } from './utils/helpers';

import {
    getUri,
    getParallel,
    getAllContainers,
    getAllNodes,
    getAllTasks,
    getAllServices,
    getAllNodeClusters,
    getWebSocket
} from './utils/request';
