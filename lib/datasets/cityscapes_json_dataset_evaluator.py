# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for evaluating results on Cityscapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import glob
import logging
import os
import sys
import uuid

import pycocotools.mask as mask_util

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import RAW_DIR

logger = logging.getLogger(__name__)


def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    if cfg.CLUSTER.ON_CLUSTER:
        # On the cluster avoid saving these files in the job directory
        output_dir = '/tmp'
    res_file = os.path.join(
        output_dir, 'segmentations_' + json_dataset.name + '_results')
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'

    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    os.environ['CITYSCAPES_DATASET'] = DATASETS[json_dataset.name][RAW_DIR]
    os.environ['CITYSCAPES_RESULTS'] = output_dir

    # Load the Cityscapes eval script *after* setting the required env vars,
    # since the script reads their values into global variables (at load time).
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling \
        as cityscapes_eval

    if not glob.glob(cityscapes_eval.args.args.groundTruthSearch):
        logger.warning("Skipping maks test since groundtruths not found by pattern " + cityscapes_eval.args.args.groundTruthSearch)
        return None
    cityscapes_eval.args.gtInstancesFile = os.path.abspath(os.path.join(
        DATASETS[json_dataset.name][RAW_DIR], "../cityscapesscripts-evaluation-cached-gtInstances.json"
        ))

    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        im_name = entry['image']

        basename = os.path.splitext(os.path.basename(im_name))[0]
        txtname = os.path.join(output_dir, basename + 'pred.txt')
        with open(txtname, 'w') as fid_txt:
            if i % 10 == 0:
                logger.info('i: {}: {}'.format(i, basename))
            for j in range(1, len(all_segms)):
                clss = json_dataset.classes[j]
                clss_id = cityscapes_eval.name2label[clss].id
                segms = all_segms[j][i]
                boxes = all_boxes[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)

                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    pngname = os.path.join(
                        'results',
                        basename + '_' + clss + '_{}.png'.format(k))
                    # write txt
                    fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                    # save mask
                    cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)
    logger.info('Evaluating...')
    argv_old = sys.argv
    sys.argv = ["force_only_unused_arg_0_workaround_for_cityscapes_eval"]
    cityscapes_eval.main()
    sys.argv = argv_old
    return None
