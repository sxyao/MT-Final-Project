__author__ = 'yuhongliang324'

import os

dn = os.path.dirname(os.path.abspath(__file__))

data_root = os.path.join(dn, 'data')
data_origin_root = '/usr0/home/hongliay/datasets/europarl'

cs_en_origin = {'czech': os.path.join(data_origin_root, 'europarl-v7.cs-en.cs'),
                'english': os.path.join(data_origin_root, 'europarl-v7.cs-en.en')}

es_en_origin = {'spanish': os.path.join(data_origin_root, 'europarl-v7.es-en.es'),
                'english': os.path.join(data_origin_root, 'europarl-v7.es-en.en')}

fr_en_origin = {'french': os.path.join(data_origin_root, ''),
                'english': os.path.join(data_origin_root, '')}

data_split_root = os.path.join(data_root, 'split')

cs_en_split = {'czech': {'train': os.path.join(data_split_root, 'cs-en.train.cs'),
                         'valid': os.path.join(data_split_root, 'cs-en.valid.cs'),
                         'test': os.path.join(data_split_root, 'cs-en.test.cs')},
               'english': {'train': os.path.join(data_split_root, 'cs-en.train.en'),
                           'valid': os.path.join(data_split_root, 'cs-en.valid.en'),
                           'test': os.path.join(data_split_root, 'cs-en.test.en')}}

es_en_split = {'spanish': {'train': os.path.join(data_split_root, 'es-en.train.de'),
                           'valid': os.path.join(data_split_root, 'es-en.valid.de'),
                           'test': os.path.join(data_split_root, 'es-en.test.de')},
               'english': {'train': os.path.join(data_split_root, 'es-en.train.en'),
                           'valid': os.path.join(data_split_root, 'es-en.valid.en'),
                           'test': os.path.join(data_split_root, 'es-en.test.en')}}

fr_en_split = {'french': {'train': os.path.join(data_split_root, 'fr-en.train.fr'),
                          'valid': os.path.join(data_split_root, 'fr-en.valid.fr'),
                          'test': os.path.join(data_split_root, 'fr-en.test.fr')},
               'english': {'train': os.path.join(data_split_root, 'fr-en.train.en'),
                           'valid': os.path.join(data_split_root, 'fr-en.valid.en'),
                           'test': os.path.join(data_split_root, 'fr-en.test.en')}}

