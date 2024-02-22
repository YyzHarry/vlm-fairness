from os.path import join


METADATA = {
    'MIMIC': join('data', 'MIMIC-CXR-JPG', 'foundation_fair_meta', 'metadata.csv'),
    'CheXpert': join('data', 'chexpert', 'foundation_fair_meta', 'metadata.csv'),
    'NIH': join('data', 'ChestXray8', 'foundation_fair_meta', 'metadata.csv'),
    'PadChest': join('data', 'PadChest', 'foundation_fair_meta', 'metadata.csv'),
    'VinDr': join('data', 'vindr-cxr', 'foundation_fair_meta', 'metadata_eval.csv')
}
METADATA_ATTR = METADATA.copy()
METADATA_ATTR['PadChest'] = join('data', 'PadChest', 'foundation_fair_meta', 'metadata_all.csv')


DATASETS = [
    'MIMIC',
    'CheXpert',
    'NIH',
    'PadChest',
    'VinDr'
]
ATTRS = ['sex', 'ethnicity', 'age', 'sex_ethnicity']

TASKS = {
    'MIMIC': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
              'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
              'Pneumothorax', 'Support Devices'],
    'CheXpert': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                 'Pneumothorax', 'Support Devices'],
    'NIH': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax'],
    'PadChest': ['adenopathy', 'air trapping', 'alveolar pattern', 'aortic atheromatosis', 'aortic button enlargement',
                 'aortic elongation', 'apical pleural thickening', 'artificial heart valve', 'atelectasis',
                 'axial hyperostosis', 'azygos lobe', 'bronchiectasis', 'bronchovascular markings', 'bullas',
                 'calcified adenopathy', 'calcified densities', 'calcified granuloma', 'calcified pleural thickening',
                 'callus rib fracture', 'cardiomegaly', 'cavitation', 'central venous catheter via jugular vein',
                 'central venous catheter via subclavian vein', 'chronic changes', 'consolidation', 'copd signs',
                 'costophrenic angle blunting', 'dai', 'descendent aortic elongation', 'diaphragmatic eventration',
                 'dual chamber device', 'emphysema', 'endotracheal tube', 'fibrotic band', 'flattened diaphragm',
                 'goiter', 'granuloma', 'ground glass pattern', 'gynecomastia', 'heart insufficiency',
                 'hemidiaphragm elevation', 'hiatal hernia', 'hilar congestion', 'hilar enlargement',
                 'hyperinflated lung', 'hypoexpansion', 'hypoexpansion basal', 'increased density', 'infiltrates',
                 'interstitial pattern', 'kyphosis', 'laminar atelectasis', 'lobar atelectasis', 'mammary prosthesis',
                 'mastectomy', 'mediastinal enlargement', 'mediastinic lipomatosis', 'metal',
                 'minor fissure thickening', 'multiple nodules', 'nipple shadow', 'nodule', 'normal', 'nsg tube',
                 'osteopenia', 'osteosynthesis material', 'pacemaker', 'pectum excavatum', 'pleural effusion',
                 'pleural thickening', 'pneumonia', 'pseudonodule', 'pulmonary fibrosis', 'pulmonary mass',
                 'rib fracture', 'sclerotic bone lesion', 'scoliosis', 'single chamber device', 'sternotomy',
                 'suboptimal study', 'superior mediastinal enlargement', 'supra aortic elongation', 'suture material',
                 'tracheal shift', 'tracheostomy tube', 'tuberculosis sequelae', 'unchanged',
                 'vascular hilar enlargement', 'vascular redistribution', 'vertebral anterior compression',
                 'vertebral compression', 'vertebral degenerative changes', 'vertebral fracture', 'volume loss'],
    'VinDr': ['Aortic enlargement', 'Atelectasis', 'COPD', 'Calcification', 'Cardiomegaly', 'Clavicle fracture',
              'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
              'Lung cavity', 'Lung cyst', 'Lung tumor', 'Mediastinal shift', 'No finding', 'Nodule/Mass',
              'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumonia', 'Pneumothorax',
              'Pulmonary fibrosis', 'Rib fracture', 'Tuberculosis']
}


ATTR_MAPPING = {
    "sex": "sex",
    "ethnicity": "ethnicity",
    "age": {"MIMIC": "anchor_age", "NIH": "Patient Age", "CheXpert": "Age",
            "PadChest": "Age", "VinDr": "age_yr"},
    "sex_ethnicity": "sex_ethnicity",
}

ATTR_NAMES = {
    "sex": ["Female", "Male"],
    "ethnicity": ["White", "Black", "Asian", "Others"],
    "age": [">80", "60~80", "40~60", "18~40", "0~18"],
    "sex_ethnicity": ["M_W", "F_W", "M_B", "F_B", "M_A", "F_A", "M_O", "F_O"],
}

SPLITS = {
    'tr': 0,
    'va': 1,
    'te': 2
}

BINARY_TEMPLATES = {
    'female': 0,  # 'male': 1,
    'white': 0, 'black': 1, 'asian': 2,
    '80-100': 0, '60-80': 1, '40-60': 2, '18-40': 3, '0-18': 4,
}
