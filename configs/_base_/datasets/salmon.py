dataset_info = dict(
    dataset_name='salmon',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='eye', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='upper_lip',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='lower_lip',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='gill',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='dorsal_fin',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='adipose_fin',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='pectoral_fin',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        7:
        dict(
            name='pelvic_fin',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        8:
        dict(
            name='anal_fin',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        9:
        dict(
            name='caudal_fin_central',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        10:
        dict(
            name='caudal_fin_upper',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        11:
        dict(
            name='caudal_fin_lower',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('eye', 'upper_lip'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('eye', 'lower_lip'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('eye', 'gill'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('upper_lip', 'lower_lip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('gill', 'dorsal_fin'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('dorsal_fin', 'adipose_fin'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('adipose_fin', 'caudal_fin_central'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('caudal_fin_central', 'anal_fin'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('anal_fin', 'pelvic_fin'), id=8, color=[0, 0, 255]),
        9:
        dict(link=('pelvic_fin', 'pectoral_fin'), id=9, color=[51, 153, 255]),
        10:
        dict(link=('pectoral_fin', 'eye'), id=10, color=[51, 153, 255]),
        11:
        dict(link=('caudal_fin_central', 'caudal_fin_upper'), id=11, color=[51, 153, 255]),
        12:
        dict(link=('caudal_fin_central', 'caudal_fin_lower'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('dorsal_fin', 'pectoral_fin'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('dorsal_fin', 'pelvic_fin'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('dorsal_fin', 'anal_fin'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('adipose_fin', 'pelvic_fin'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('adipose_fin', 'anal_fin'), id=17, color=[51, 153, 255]),
        18:
        dict(link=('eye', 'dorsal_fin'), id=18, color=[51, 153, 255]),

    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1.,],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107
    ])
# TODO: make skeleton colorful
import matplotlib.cm as cm
def get_color_from_id(id_, max_id=17, color_map='jet'):
    x = float(id_/max_id)
    color = getattr(cm, color_map)(x)
    return [color[0] * 255, color[1] * 255, color[2] * 255]

name2id = {}
for key in dataset_info['keypoint_info'].keys():
    id_ = dataset_info['keypoint_info'][key]['id']
    max_id = len(dataset_info['keypoint_info'].keys())
    dataset_info['keypoint_info'][key]['color'] = get_color_from_id(id_, max_id, 'jet')
    name = dataset_info['keypoint_info'][key]['name']
    name2id[name] = id_

for key in dataset_info['skeleton_info'].keys():
    name1, name2 = dataset_info['skeleton_info'][key]['link']
    id1 = name2id[name1]
    id2 = name2id[name2]
    color1 = dataset_info['keypoint_info'][id1]['color']
    color2 = dataset_info['keypoint_info'][id2]['color']
    color = [int((c1+c2)/2) for c1,c2 in zip(color1, color2)]
    dataset_info['skeleton_info'][key]['color'] = color


    