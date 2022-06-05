# # dataset settings
# dataset_type = 'CocoDataset'
# data_root = '/data/vision/polina/scratch/clintonw/datasets/coco/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.json',
#         img_prefix=data_root + 'train2017/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')
# __all__ = ['CocoPanopticDataset']

# # A custom value to distinguish instance ID and category ID; need to
# # be greater than the number of categories.
# # For a pixel in the panoptic result map:
# #   pan_id = ins_id * INSTANCE_OFFSET + cat_id
# INSTANCE_OFFSET = 1000


# class COCOPanoptic(COCO):
#     """This wrapper is for loading the panoptic style annotation file.

#     The format is shown in the CocoPanopticDataset class.

#     Args:
#         annotation_file (str): Path of annotation file.
#     """

#     def __init__(self, annotation_file=None):
#         if panopticapi is None:
#             raise RuntimeError(
#                 'panopticapi is not installed, please install it by: '
#                 'pip install git+https://github.com/cocodataset/panopticapi.git.')

#         super(COCOPanoptic, self).__init__(annotation_file)

#     def createIndex(self):
#         # create index
#         print('creating index...')
#         # anns stores 'segment_id -> annotation'
#         anns, cats, imgs = {}, {}, {}
#         img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
#         if 'annotations' in self.dataset:
#             for ann, img_info in zip(self.dataset['annotations'],
#                                      self.dataset['images']):
#                 img_info['segm_file'] = ann['file_name']
#                 for seg_ann in ann['segments_info']:
#                     # to match with instance.json
#                     seg_ann['image_id'] = ann['image_id']
#                     seg_ann['height'] = img_info['height']
#                     seg_ann['width'] = img_info['width']
#                     img_to_anns[ann['image_id']].append(seg_ann)
#                     # segment_id is not unique in coco dataset orz...
#                     if seg_ann['id'] in anns.keys():
#                         anns[seg_ann['id']].append(seg_ann)
#                     else:
#                         anns[seg_ann['id']] = [seg_ann]

#         if 'images' in self.dataset:
#             for img in self.dataset['images']:
#                 imgs[img['id']] = img

#         if 'categories' in self.dataset:
#             for cat in self.dataset['categories']:
#                 cats[cat['id']] = cat

#         if 'annotations' in self.dataset and 'categories' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 for seg_ann in ann['segments_info']:
#                     cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

#         print('index created!')

#         self.anns = anns
#         self.imgToAnns = img_to_anns
#         self.catToImgs = cat_to_imgs
#         self.imgs = imgs
#         self.cats = cats

#     def load_anns(self, ids=[]):
#         """Load anns with the specified ids.

#         self.anns is a list of annotation lists instead of a
#         list of annotations.

#         Args:
#             ids (int array): integer ids specifying anns

#         Returns:
#             anns (object array): loaded ann objects
#         """
#         anns = []

#         if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
#             # self.anns is a list of annotation lists instead of
#             # a list of annotations
#             for id in ids:
#                 anns += self.anns[id]
#             return anns
#         elif type(ids) == int:
#             return self.anns[ids]


# @DATASETS.register_module()
# class CocoPanopticDataset(CocoDataset):
#     """Coco dataset for Panoptic segmentation.

#     The annotation format is shown as follows. The `ann` field is optional
#     for testing.

#     .. code-block:: none

#         [
#             {
#                 'filename': f'{image_id:012}.png',
#                 'image_id':9
#                 'segments_info': {
#                     [
#                         {
#                             'id': 8345037, (segment_id in panoptic png,
#                                             convert from rgb)
#                             'category_id': 51,
#                             'iscrowd': 0,
#                             'bbox': (x1, y1, w, h),
#                             'area': 24315,
#                             'segmentation': list,(coded mask)
#                         },
#                         ...
#                     }
#                 }
#             },
#             ...
#         ]
#     """
#     CLASSES = [
#         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#         ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#         'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
#         'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
#         'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
#         'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
#         'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
#         'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
#         'wall-wood', 'water-other', 'window-blind', 'window-other',
#         'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
#         'cabinet-merged', 'table-merged', 'floor-other-merged',
#         'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
#         'paper-merged', 'food-other-merged', 'building-other-merged',
#         'rock-merged', 'wall-other-merged', 'rug-merged'
#     ]
#     THING_CLASSES = [
#         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#         'scissors', 'teddy bear', 'hair drier', 'toothbrush'
#     ]
#     STUFF_CLASSES = [
#         'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
#         'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
#         'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
#         'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
#         'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
#         'wall-wood', 'water-other', 'window-blind', 'window-other',
#         'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
#         'cabinet-merged', 'table-merged', 'floor-other-merged',
#         'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
#         'paper-merged', 'food-other-merged', 'building-other-merged',
#         'rock-merged', 'wall-other-merged', 'rug-merged'
#     ]

#     def load_annotations(self, ann_file):
#         """Load annotation from COCO Panoptic style annotation file.

#         Args:
#             ann_file (str): Path of annotation file.

#         Returns:
#             list[dict]: Annotation info from COCO api.
#         """
#         self.coco = COCOPanoptic(ann_file)
#         self.cat_ids = self.coco.get_cat_ids()
#         self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
#         self.categories = self.coco.cats
#         self.img_ids = self.coco.get_img_ids()
#         data_infos = []
#         for i in self.img_ids:
#             info = self.coco.load_imgs([i])[0]
#             info['filename'] = info['file_name']
#             info['segm_file'] = info['filename'].replace('jpg', 'png')
#             data_infos.append(info)
#         return data_infos

#     def get_ann_info(self, idx):
#         """Get COCO annotation by index.

#         Args:
#             idx (int): Index of data.

#         Returns:
#             dict: Annotation info of specified index.
#         """
#         img_id = self.data_infos[idx]['id']
#         ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#         ann_info = self.coco.load_anns(ann_ids)
#         # filter out unmatched images
#         ann_info = [i for i in ann_info if i['image_id'] == img_id]
#         return self._parse_ann_info(self.data_infos[idx], ann_info)

#     def _parse_ann_info(self, img_info, ann_info):
#         """Parse annotations and load panoptic ground truths.

#         Args:
#             img_info (int): Image info of an image.
#             ann_info (list[dict]): Annotation info of an image.

#         Returns:
#             dict: A dict containing the following keys: bboxes, bboxes_ignore,
#                 labels, masks, seg_map.
#         """
#         gt_bboxes = []
#         gt_labels = []
#         gt_bboxes_ignore = []
#         gt_mask_infos = []

#         for i, ann in enumerate(ann_info):
#             x1, y1, w, h = ann['bbox']
#             if ann['area'] <= 0 or w < 1 or h < 1:
#                 continue
#             bbox = [x1, y1, x1 + w, y1 + h]

#             category_id = ann['category_id']
#             contiguous_cat_id = self.cat2label[category_id]

#             is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
#             if is_thing:
#                 is_crowd = ann.get('iscrowd', False)
#                 if not is_crowd:
#                     gt_bboxes.append(bbox)
#                     gt_labels.append(contiguous_cat_id)
#                 else:
#                     gt_bboxes_ignore.append(bbox)
#                     is_thing = False

#             mask_info = {
#                 'id': ann['id'],
#                 'category': contiguous_cat_id,
#                 'is_thing': is_thing
#             }
#             gt_mask_infos.append(mask_info)

#         if gt_bboxes:
#             gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
#             gt_labels = np.array(gt_labels, dtype=np.int64)
#         else:
#             gt_bboxes = np.zeros((0, 4), dtype=np.float32)
#             gt_labels = np.array([], dtype=np.int64)

#         if gt_bboxes_ignore:
#             gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
#         else:
#             gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

#         ann = dict(
#             bboxes=gt_bboxes,
#             labels=gt_labels,
#             bboxes_ignore=gt_bboxes_ignore,
#             masks=gt_mask_infos,
#             seg_map=img_info['segm_file'])

#         return ann

#     def _filter_imgs(self, min_size=32):
#         """Filter images too small or without ground truths."""
#         ids_with_ann = []
#         # check whether images have legal thing annotations.
#         for lists in self.coco.anns.values():
#             for item in lists:
#                 category_id = item['category_id']
#                 is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
#                 if not is_thing:
#                     continue
#                 ids_with_ann.append(item['image_id'])
#         ids_with_ann = set(ids_with_ann)

#         valid_inds = []
#         valid_img_ids = []
#         for i, img_info in enumerate(self.data_infos):
#             img_id = self.img_ids[i]
#             if self.filter_empty_gt and img_id not in ids_with_ann:
#                 continue
#             if min(img_info['width'], img_info['height']) >= min_size:
#                 valid_inds.append(i)
#                 valid_img_ids.append(img_id)
#         self.img_ids = valid_img_ids
#         return valid_inds

#     def _pan2json(self, results, outfile_prefix):
#         """Convert panoptic results to COCO panoptic json style."""
#         label2cat = dict((v, k) for (k, v) in self.cat2label.items())
#         pred_annotations = []
#         outdir = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

#         for idx in range(len(self)):
#             img_id = self.img_ids[idx]
#             segm_file = self.data_infos[idx]['segm_file']
#             pan = results[idx]

#             pan_labels = np.unique(pan)
#             segm_info = []
#             for pan_label in pan_labels:
#                 sem_label = pan_label % INSTANCE_OFFSET
#                 # We reserve the length of self.CLASSES for VOID label
#                 if sem_label == len(self.CLASSES):
#                     continue
#                 # convert sem_label to json label
#                 cat_id = label2cat[sem_label]
#                 is_thing = self.categories[cat_id]['isthing']
#                 mask = pan == pan_label
#                 area = mask.sum()
#                 segm_info.append({
#                     'id': int(pan_label),
#                     'category_id': cat_id,
#                     'isthing': is_thing,
#                     'area': int(area)
#                 })
#             # evaluation script uses 0 for VOID label.
#             pan[pan % INSTANCE_OFFSET == len(self.CLASSES)] = VOID
#             pan = id2rgb(pan).astype(np.uint8)
#             mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
#             record = {
#                 'image_id': img_id,
#                 'segments_info': segm_info,
#                 'file_name': segm_file
#             }
#             pred_annotations.append(record)
#         pan_json_results = dict(annotations=pred_annotations)
#         return pan_json_results

#     def results2json(self, results, outfile_prefix):
#         """Dump the panoptic results to a COCO panoptic style json file.

#         Args:
#             results (dict): Testing results of the dataset.
#             outfile_prefix (str): The filename prefix of the json files. If the
#                 prefix is "somepath/xxx", the json files will be named
#                 "somepath/xxx.panoptic.json"

#         Returns:
#             dict[str: str]: The key is 'panoptic' and the value is
#                 corresponding filename.
#         """
#         result_files = dict()
#         pan_results = [result['pan_results'] for result in results]
#         pan_json_results = self._pan2json(pan_results, outfile_prefix)
#         result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
#         mmcv.dump(pan_json_results, result_files['panoptic'])

#         return result_files

#     def evaluate_pan_json(self,
#                           result_files,
#                           outfile_prefix,
#                           logger=None,
#                           classwise=False):
#         """Evaluate PQ according to the panoptic results json file."""
#         imgs = self.coco.imgs
#         gt_json = self.coco.img_ann_map  # image to annotations
#         gt_json = [{
#             'image_id': k,
#             'segments_info': v,
#             'file_name': imgs[k]['segm_file']
#         } for k, v in gt_json.items()]
#         pred_json = mmcv.load(result_files['panoptic'])
#         pred_json = dict(
#             (el['image_id'], el) for el in pred_json['annotations'])

#         # match the gt_anns and pred_anns in the same image
#         matched_annotations_list = []
#         for gt_ann in gt_json:
#             img_id = gt_ann['image_id']
#             if img_id not in pred_json.keys():
#                 raise Exception('no prediction for the image'
#                                 ' with id: {}'.format(img_id))
#             matched_annotations_list.append((gt_ann, pred_json[img_id]))

#         gt_folder = self.seg_prefix
#         pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

#         pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder,
#                                         pred_folder, self.categories,
#                                         self.file_client)

#         metrics = [('All', None), ('Things', True), ('Stuff', False)]
#         pq_results = {}

#         for name, isthing in metrics:
#             pq_results[name], classwise_results = pq_stat.pq_average(
#                 self.categories, isthing=isthing)
#             if name == 'All':
#                 pq_results['classwise'] = classwise_results

#         classwise_results = None
#         if classwise:
#             classwise_results = {
#                 k: v
#                 for k, v in zip(self.CLASSES, pq_results['classwise'].values())
#             }
#         print_panoptic_table(pq_results, classwise_results, logger=logger)

#         return parse_pq_results(pq_results)

#     def evaluate(self,
#                  results,
#                  metric='PQ',
#                  logger=None,
#                  jsonfile_prefix=None,
#                  classwise=False,
#                  **kwargs):
#         """Evaluation in COCO Panoptic protocol.

#         Args:
#             results (list[dict]): Testing results of the dataset.
#             metric (str | list[str]): Metrics to be evaluated. Only
#                 support 'PQ' at present. 'pq' will be regarded as 'PQ.
#             logger (logging.Logger | str | None): Logger used for printing
#                 related information during evaluation. Default: None.
#             jsonfile_prefix (str | None): The prefix of json files. It includes
#                 the file path and the prefix of filename, e.g., "a/b/prefix".
#                 If not specified, a temp file will be created. Default: None.
#             classwise (bool): Whether to print classwise evaluation results.
#                 Default: False.

#         Returns:
#             dict[str, float]: COCO Panoptic style evaluation metric.
#         """
#         metrics = metric if isinstance(metric, list) else [metric]
#         # Compatible with lowercase 'pq'
#         metrics = ['PQ' if metric == 'pq' else metric for metric in metrics]
#         allowed_metrics = ['PQ']  # todo: support other metrics like 'bbox'
#         for metric in metrics:
#             if metric not in allowed_metrics:
#                 raise KeyError(f'metric {metric} is not supported')

#         result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
#         eval_results = {}

#         outfile_prefix = os.path.join(tmp_dir.name, 'results') \
#             if tmp_dir is not None else jsonfile_prefix
#         if 'PQ' in metrics:
#             eval_pan_results = self.evaluate_pan_json(result_files,
#                                                       outfile_prefix, logger,
#                                                       classwise)
#             eval_results.update(eval_pan_results)

#         if tmp_dir is not None:
#             tmp_dir.cleanup()
#         return eval_results


# def parse_pq_results(pq_results):
#     """Parse the Panoptic Quality results."""
#     result = dict()
#     result['PQ'] = 100 * pq_results['All']['pq']
#     result['SQ'] = 100 * pq_results['All']['sq']
#     result['RQ'] = 100 * pq_results['All']['rq']
#     result['PQ_th'] = 100 * pq_results['Things']['pq']
#     result['SQ_th'] = 100 * pq_results['Things']['sq']
#     result['RQ_th'] = 100 * pq_results['Things']['rq']
#     result['PQ_st'] = 100 * pq_results['Stuff']['pq']
#     result['SQ_st'] = 100 * pq_results['Stuff']['sq']
#     result['RQ_st'] = 100 * pq_results['Stuff']['rq']
#     return result


# def print_panoptic_table(pq_results, classwise_results=None, logger=None):
#     """Print the panoptic evaluation results table.

#     Args:
#         pq_results(dict): The Panoptic Quality results.
#         classwise_results(dict | None): The classwise Panoptic Quality results.
#             The keys are class names and the values are metrics.
#         logger (logging.Logger | str | None): Logger used for printing
#             related information during evaluation. Default: None.
#     """

#     headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
#     data = [headers]
#     for name in ['All', 'Things', 'Stuff']:
#         numbers = [
#             f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
#         ]
#         row = [name] + numbers + [pq_results[name]['n']]
#         data.append(row)
#     table = AsciiTable(data)
#     print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)

#     if classwise_results is not None:
#         class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
#                                           for k in ['pq', 'sq', 'rq'])
#                          for name, metrics in classwise_results.items()]
#         num_columns = min(8, len(class_metrics) * 4)
#         results_flatten = list(itertools.chain(*class_metrics))
#         headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
#         results_2d = itertools.zip_longest(
#             *[results_flatten[i::num_columns] for i in range(num_columns)])
#         data = [headers]
#         data += [result for result in results_2d]
#         table = AsciiTable(data)
#         print_log(
#             'Classwise Panoptic Evaluation Results:\n' + table.table,
#             logger=logger)
# # Copyright (c) OpenMMLab. All rights reserved.


# class CocoDataset(CustomDataset):

#     CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

#     def load_annotations(self, ann_file):
#         """Load annotation from COCO style annotation file.

#         Args:
#             ann_file (str): Path of annotation file.

#         Returns:
#             list[dict]: Annotation info from COCO api.
#         """

#         self.coco = COCO(ann_file)
#         # The order of returned `cat_ids` will not
#         # change with the order of the CLASSES
#         self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

#         self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
#         self.img_ids = self.coco.get_img_ids()
#         data_infos = []
#         total_ann_ids = []
#         for i in self.img_ids:
#             info = self.coco.load_imgs([i])[0]
#             info['filename'] = info['file_name']
#             data_infos.append(info)
#             ann_ids = self.coco.get_ann_ids(img_ids=[i])
#             total_ann_ids.extend(ann_ids)
#         assert len(set(total_ann_ids)) == len(
#             total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
#         return data_infos

#     def get_ann_info(self, idx):
#         """Get COCO annotation by index.

#         Args:
#             idx (int): Index of data.

#         Returns:
#             dict: Annotation info of specified index.
#         """

#         img_id = self.data_infos[idx]['id']
#         ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#         ann_info = self.coco.load_anns(ann_ids)
#         return self._parse_ann_info(self.data_infos[idx], ann_info)

#     def get_cat_ids(self, idx):
#         """Get COCO category ids by index.

#         Args:
#             idx (int): Index of data.

#         Returns:
#             list[int]: All categories in the image of specified index.
#         """

#         img_id = self.data_infos[idx]['id']
#         ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#         ann_info = self.coco.load_anns(ann_ids)
#         return [ann['category_id'] for ann in ann_info]

#     def _filter_imgs(self, min_size=32):
#         """Filter images too small or without ground truths."""
#         valid_inds = []
#         # obtain images that contain annotation
#         ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
#         # obtain images that contain annotations of the required categories
#         ids_in_cat = set()
#         for i, class_id in enumerate(self.cat_ids):
#             ids_in_cat |= set(self.coco.cat_img_map[class_id])
#         # merge the image id sets of the two conditions and use the merged set
#         # to filter out images if self.filter_empty_gt=True
#         ids_in_cat &= ids_with_ann

#         valid_img_ids = []
#         for i, img_info in enumerate(self.data_infos):
#             img_id = self.img_ids[i]
#             if self.filter_empty_gt and img_id not in ids_in_cat:
#                 continue
#             if min(img_info['width'], img_info['height']) >= min_size:
#                 valid_inds.append(i)
#                 valid_img_ids.append(img_id)
#         self.img_ids = valid_img_ids
#         return valid_inds

#     def _parse_ann_info(self, img_info, ann_info):
#         """Parse bbox and mask annotation.

#         Args:
#             ann_info (list[dict]): Annotation info of an image.
#             with_mask (bool): Whether to parse mask annotations.

#         Returns:
#             dict: A dict containing the following keys: bboxes, bboxes_ignore,\
#                 labels, masks, seg_map. "masks" are raw annotations and not \
#                 decoded into binary masks.
#         """
#         gt_bboxes = []
#         gt_labels = []
#         gt_bboxes_ignore = []
#         gt_masks_ann = []
#         for i, ann in enumerate(ann_info):
#             if ann.get('ignore', False):
#                 continue
#             x1, y1, w, h = ann['bbox']
#             inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
#             inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
#             if inter_w * inter_h == 0:
#                 continue
#             if ann['area'] <= 0 or w < 1 or h < 1:
#                 continue
#             if ann['category_id'] not in self.cat_ids:
#                 continue
#             bbox = [x1, y1, x1 + w, y1 + h]
#             if ann.get('iscrowd', False):
#                 gt_bboxes_ignore.append(bbox)
#             else:
#                 gt_bboxes.append(bbox)
#                 gt_labels.append(self.cat2label[ann['category_id']])
#                 gt_masks_ann.append(ann.get('segmentation', None))

#         if gt_bboxes:
#             gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
#             gt_labels = np.array(gt_labels, dtype=np.int64)
#         else:
#             gt_bboxes = np.zeros((0, 4), dtype=np.float32)
#             gt_labels = np.array([], dtype=np.int64)

#         if gt_bboxes_ignore:
#             gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
#         else:
#             gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

#         seg_map = img_info['filename'].replace('jpg', 'png')

#         ann = dict(
#             bboxes=gt_bboxes,
#             labels=gt_labels,
#             bboxes_ignore=gt_bboxes_ignore,
#             masks=gt_masks_ann,
#             seg_map=seg_map)

#         return ann

#     def xyxy2xywh(self, bbox):
#         """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
#         evaluation.

#         Args:
#             bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
#                 ``xyxy`` order.

#         Returns:
#             list[float]: The converted bounding boxes, in ``xywh`` order.
#         """

#         _bbox = bbox.tolist()
#         return [
#             _bbox[0],
#             _bbox[1],
#             _bbox[2] - _bbox[0],
#             _bbox[3] - _bbox[1],
#         ]

#     def _proposal2json(self, results):
#         """Convert proposal results to COCO json style."""
#         json_results = []
#         for idx in range(len(self)):
#             img_id = self.img_ids[idx]
#             bboxes = results[idx]
#             for i in range(bboxes.shape[0]):
#                 data = dict()
#                 data['image_id'] = img_id
#                 data['bbox'] = self.xyxy2xywh(bboxes[i])
#                 data['score'] = float(bboxes[i][4])
#                 data['category_id'] = 1
#                 json_results.append(data)
#         return json_results

#     def _det2json(self, results):
#         """Convert detection results to COCO json style."""
#         json_results = []
#         for idx in range(len(self)):
#             img_id = self.img_ids[idx]
#             result = results[idx]
#             for label in range(len(result)):
#                 bboxes = result[label]
#                 for i in range(bboxes.shape[0]):
#                     data = dict()
#                     data['image_id'] = img_id
#                     data['bbox'] = self.xyxy2xywh(bboxes[i])
#                     data['score'] = float(bboxes[i][4])
#                     data['category_id'] = self.cat_ids[label]
#                     json_results.append(data)
#         return json_results

#     def _segm2json(self, results):
#         """Convert instance segmentation results to COCO json style."""
#         bbox_json_results = []
#         segm_json_results = []
#         for idx in range(len(self)):
#             img_id = self.img_ids[idx]
#             det, seg = results[idx]
#             for label in range(len(det)):
#                 # bbox results
#                 bboxes = det[label]
#                 for i in range(bboxes.shape[0]):
#                     data = dict()
#                     data['image_id'] = img_id
#                     data['bbox'] = self.xyxy2xywh(bboxes[i])
#                     data['score'] = float(bboxes[i][4])
#                     data['category_id'] = self.cat_ids[label]
#                     bbox_json_results.append(data)

#                 # segm results
#                 # some detectors use different scores for bbox and mask
#                 if isinstance(seg, tuple):
#                     segms = seg[0][label]
#                     mask_score = seg[1][label]
#                 else:
#                     segms = seg[label]
#                     mask_score = [bbox[4] for bbox in bboxes]
#                 for i in range(bboxes.shape[0]):
#                     data = dict()
#                     data['image_id'] = img_id
#                     data['bbox'] = self.xyxy2xywh(bboxes[i])
#                     data['score'] = float(mask_score[i])
#                     data['category_id'] = self.cat_ids[label]
#                     if isinstance(segms[i]['counts'], bytes):
#                         segms[i]['counts'] = segms[i]['counts'].decode()
#                     data['segmentation'] = segms[i]
#                     segm_json_results.append(data)
#         return bbox_json_results, segm_json_results

#     def results2json(self, results, outfile_prefix):
#         """Dump the detection results to a COCO style json file.

#         There are 3 types of results: proposals, bbox predictions, mask
#         predictions, and they have different data types. This method will
#         automatically recognize the type, and dump them to json files.

#         Args:
#             results (list[list | tuple | ndarray]): Testing results of the
#                 dataset.
#             outfile_prefix (str): The filename prefix of the json files. If the
#                 prefix is "somepath/xxx", the json files will be named
#                 "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
#                 "somepath/xxx.proposal.json".

#         Returns:
#             dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
#                 values are corresponding filenames.
#         """
#         result_files = dict()
#         if isinstance(results[0], list):
#             json_results = self._det2json(results)
#             result_files['bbox'] = f'{outfile_prefix}.bbox.json'
#             result_files['proposal'] = f'{outfile_prefix}.bbox.json'
#             mmcv.dump(json_results, result_files['bbox'])
#         elif isinstance(results[0], tuple):
#             json_results = self._segm2json(results)
#             result_files['bbox'] = f'{outfile_prefix}.bbox.json'
#             result_files['proposal'] = f'{outfile_prefix}.bbox.json'
#             result_files['segm'] = f'{outfile_prefix}.segm.json'
#             mmcv.dump(json_results[0], result_files['bbox'])
#             mmcv.dump(json_results[1], result_files['segm'])
#         elif isinstance(results[0], np.ndarray):
#             json_results = self._proposal2json(results)
#             result_files['proposal'] = f'{outfile_prefix}.proposal.json'
#             mmcv.dump(json_results, result_files['proposal'])
#         else:
#             raise TypeError('invalid type of results')
#         return result_files

#     def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
#         gt_bboxes = []
#         for i in range(len(self.img_ids)):
#             ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
#             ann_info = self.coco.load_anns(ann_ids)
#             if len(ann_info) == 0:
#                 gt_bboxes.append(np.zeros((0, 4)))
#                 continue
#             bboxes = []
#             for ann in ann_info:
#                 if ann.get('ignore', False) or ann['iscrowd']:
#                     continue
#                 x1, y1, w, h = ann['bbox']
#                 bboxes.append([x1, y1, x1 + w, y1 + h])
#             bboxes = np.array(bboxes, dtype=np.float32)
#             if bboxes.shape[0] == 0:
#                 bboxes = np.zeros((0, 4))
#             gt_bboxes.append(bboxes)

#         recalls = eval_recalls(
#             gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
#         ar = recalls.mean(axis=1)
#         return ar

#     def format_results(self, results, jsonfile_prefix=None, **kwargs):
#         """Format the results to json (standard format for COCO evaluation).

#         Args:
#             results (list[tuple | numpy.ndarray]): Testing results of the
#                 dataset.
#             jsonfile_prefix (str | None): The prefix of json files. It includes
#                 the file path and the prefix of filename, e.g., "a/b/prefix".
#                 If not specified, a temp file will be created. Default: None.

#         Returns:
#             tuple: (result_files, tmp_dir), result_files is a dict containing \
#                 the json filepaths, tmp_dir is the temporal directory created \
#                 for saving json files when jsonfile_prefix is not specified.
#         """
#         assert isinstance(results, list), 'results must be a list'
#         assert len(results) == len(self), (
#             'The length of results is not equal to the dataset len: {} != {}'.
#             format(len(results), len(self)))

#         if jsonfile_prefix is None:
#             tmp_dir = tempfile.TemporaryDirectory()
#             jsonfile_prefix = osp.join(tmp_dir.name, 'results')
#         else:
#             tmp_dir = None
#         result_files = self.results2json(results, jsonfile_prefix)
#         return result_files, tmp_dir

#     def evaluate(self,
#                  results,
#                  metric='bbox',
#                  logger=None,
#                  jsonfile_prefix=None,
#                  classwise=False,
#                  proposal_nums=(100, 300, 1000),
#                  iou_thrs=None,
#                  metric_items=None):
#         """Evaluation in COCO protocol.

#         Args:
#             results (list[list | tuple]): Testing results of the dataset.
#             metric (str | list[str]): Metrics to be evaluated. Options are
#                 'bbox', 'segm', 'proposal', 'proposal_fast'.
#             logger (logging.Logger | str | None): Logger used for printing
#                 related information during evaluation. Default: None.
#             jsonfile_prefix (str | None): The prefix of json files. It includes
#                 the file path and the prefix of filename, e.g., "a/b/prefix".
#                 If not specified, a temp file will be created. Default: None.
#             classwise (bool): Whether to evaluating the AP for each class.
#             proposal_nums (Sequence[int]): Proposal number used for evaluating
#                 recalls, such as recall@100, recall@1000.
#                 Default: (100, 300, 1000).
#             iou_thrs (Sequence[float], optional): IoU threshold used for
#                 evaluating recalls/mAPs. If set to a list, the average of all
#                 IoUs will also be computed. If not specified, [0.50, 0.55,
#                 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
#                 Default: None.
#             metric_items (list[str] | str, optional): Metric items that will
#                 be returned. If not specified, ``['AR@100', 'AR@300',
#                 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
#                 used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
#                 'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
#                 ``metric=='bbox' or metric=='segm'``.

#         Returns:
#             dict[str, float]: COCO style evaluation metric.
#         """

#         metrics = metric if isinstance(metric, list) else [metric]
#         allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
#         for metric in metrics:
#             if metric not in allowed_metrics:
#                 raise KeyError(f'metric {metric} is not supported')
#         if iou_thrs is None:
#             iou_thrs = np.linspace(
#                 .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
#         if metric_items is not None:
#             if not isinstance(metric_items, list):
#                 metric_items = [metric_items]

#         result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

#         eval_results = OrderedDict()
#         cocoGt = self.coco
#         for metric in metrics:
#             msg = f'Evaluating {metric}...'
#             if logger is None:
#                 msg = '\n' + msg
#             print_log(msg, logger=logger)

#             if metric == 'proposal_fast':
#                 ar = self.fast_eval_recall(
#                     results, proposal_nums, iou_thrs, logger='silent')
#                 log_msg = []
#                 for i, num in enumerate(proposal_nums):
#                     eval_results[f'AR@{num}'] = ar[i]
#                     log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
#                 log_msg = ''.join(log_msg)
#                 print_log(log_msg, logger=logger)
#                 continue

#             iou_type = 'bbox' if metric == 'proposal' else metric
#             if metric not in result_files:
#                 raise KeyError(f'{metric} is not in results')
#             try:
#                 predictions = mmcv.load(result_files[metric])
#                 if iou_type == 'segm':
#                     # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
#                     # When evaluating mask AP, if the results contain bbox,
#                     # cocoapi will use the box area instead of the mask area
#                     # for calculating the instance area. Though the overall AP
#                     # is not affected, this leads to different
#                     # small/medium/large mask AP results.
#                     for x in predictions:
#                         x.pop('bbox')
#                     warnings.simplefilter('once')
#                     warnings.warn(
#                         'The key "bbox" is deleted for more accurate mask AP '
#                         'of small/medium/large instances since v2.12.0. This '
#                         'does not change the overall mAP calculation.',
#                         UserWarning)
#                 cocoDt = cocoGt.loadRes(predictions)
#             except IndexError:
#                 print_log(
#                     'The testing results of the whole dataset is empty.',
#                     logger=logger,
#                     level=logging.ERROR)
#                 break

#             cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
#             cocoEval.params.catIds = self.cat_ids
#             cocoEval.params.imgIds = self.img_ids
#             cocoEval.params.maxDets = list(proposal_nums)
#             cocoEval.params.iouThrs = iou_thrs
#             # mapping of cocoEval.stats
#             coco_metric_names = {
#                 'mAP': 0,
#                 'mAP_50': 1,
#                 'mAP_75': 2,
#                 'mAP_s': 3,
#                 'mAP_m': 4,
#                 'mAP_l': 5,
#                 'AR@100': 6,
#                 'AR@300': 7,
#                 'AR@1000': 8,
#                 'AR_s@1000': 9,
#                 'AR_m@1000': 10,
#                 'AR_l@1000': 11
#             }
#             if metric_items is not None:
#                 for metric_item in metric_items:
#                     if metric_item not in coco_metric_names:
#                         raise KeyError(
#                             f'metric item {metric_item} is not supported')

#             if metric == 'proposal':
#                 cocoEval.params.useCats = 0
#                 cocoEval.evaluate()
#                 cocoEval.accumulate()

#                 # Save coco summarize print information to logger
#                 redirect_string = io.StringIO()
#                 with contextlib.redirect_stdout(redirect_string):
#                     cocoEval.summarize()
#                 print_log('\n' + redirect_string.getvalue(), logger=logger)

#                 if metric_items is None:
#                     metric_items = [
#                         'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
#                         'AR_m@1000', 'AR_l@1000'
#                     ]

#                 for item in metric_items:
#                     val = float(
#                         f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
#                     eval_results[item] = val
#             else:
#                 cocoEval.evaluate()
#                 cocoEval.accumulate()

#                 # Save coco summarize print information to logger
#                 redirect_string = io.StringIO()
#                 with contextlib.redirect_stdout(redirect_string):
#                     cocoEval.summarize()
#                 print_log('\n' + redirect_string.getvalue(), logger=logger)

#                 if classwise:  # Compute per-category AP
#                     # Compute per-category AP
#                     # from https://github.com/facebookresearch/detectron2/
#                     precisions = cocoEval.eval['precision']
#                     # precision: (iou, recall, cls, area range, max dets)
#                     assert len(self.cat_ids) == precisions.shape[2]

#                     results_per_category = []
#                     for idx, catId in enumerate(self.cat_ids):
#                         # area range index 0: all area ranges
#                         # max dets index -1: typically 100 per image
#                         nm = self.coco.loadCats(catId)[0]
#                         precision = precisions[:, :, idx, 0, -1]
#                         precision = precision[precision > -1]
#                         if precision.size:
#                             ap = np.mean(precision)
#                         else:
#                             ap = float('nan')
#                         results_per_category.append(
#                             (f'{nm["name"]}', f'{float(ap):0.3f}'))

#                     num_columns = min(6, len(results_per_category) * 2)
#                     results_flatten = list(
#                         itertools.chain(*results_per_category))
#                     headers = ['category', 'AP'] * (num_columns // 2)
#                     results_2d = itertools.zip_longest(*[
#                         results_flatten[i::num_columns]
#                         for i in range(num_columns)
#                     ])
#                     table_data = [headers]
#                     table_data += [result for result in results_2d]
#                     table = AsciiTable(table_data)
#                     print_log('\n' + table.table, logger=logger)

#                 if metric_items is None:
#                     metric_items = [
#                         'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
#                     ]

#                 for metric_item in metric_items:
#                     key = f'{metric}_{metric_item}'
#                     val = float(
#                         f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
#                     )
#                     eval_results[key] = val
#                 ap = cocoEval.stats[:6]
#                 eval_results[f'{metric}_mAP_copypaste'] = (
#                     f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
#                     f'{ap[4]:.3f} {ap[5]:.3f}')
#         if tmp_dir is not None:
#             tmp_dir.cleanup()
#         return eval_results
