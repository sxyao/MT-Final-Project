__author__ = 'yuhongliang324'

import os

dn = os.path.dirname(os.path.abspath(__file__))

data_root = os.path.join(dn, 'data')
data_origin_root = os.path.join(data_root, 'origin')

cs_en_origin = {'czech': os.path.join(data_origin_root, 'news-commentary-v9.cs-en.cs'),
                'english': os.path.join(data_origin_root, 'news-commentary-v9.cs-en.en')}

de_en_origin = {'german': os.path.join(data_origin_root, 'news-commentary-v9.de-en.de'),
                'english': os.path.join(data_origin_root, 'news-commentary-v9.de-en.en')}

fr_en_origin = {'french': os.path.join(data_origin_root, 'news-commentary-v9.fr-en.fr'),
                'english': os.path.join(data_origin_root, 'news-commentary-v9.fr-en.en')}

ru_en_origin = {'russian': os.path.join(data_origin_root, 'news-commentary-v9.ru-en.ru'),
                'english': os.path.join(data_origin_root, 'news-commentary-v9.ru-en.en')}

data_split_root = os.path.join(data_root, 'split')

cs_en_split = {'czech': {'train': os.path.join(data_split_root, 'cs-en.train.cs'),
                         'valid': os.path.join(data_split_root, 'cs-en.valid.cs'),
                         'test': os.path.join(data_split_root, 'cs-en.test.cs')},
               'english': {'train': os.path.join(data_split_root, 'cs-en.train.en'),
                           'valid': os.path.join(data_split_root, 'cs-en.valid.en'),
                           'test': os.path.join(data_split_root, 'cs-en.test.en')}}

de_en_split = {'german': {'train': os.path.join(data_split_root, 'de-en.train.de'),
                          'valid': os.path.join(data_split_root, 'de-en.valid.de'),
                          'test': os.path.join(data_split_root, 'de-en.test.de')},
               'english': {'train': os.path.join(data_split_root, 'de-en.train.en'),
                           'valid': os.path.join(data_split_root, 'de-en.valid.en'),
                           'test': os.path.join(data_split_root, 'de-en.test.en')}}

fr_en_split = {'french': {'train': os.path.join(data_split_root, 'fr-en.train.fr'),
                          'valid': os.path.join(data_split_root, 'fr-en.valid.fr'),
                          'test': os.path.join(data_split_root, 'fr-en.test.fr')},
               'english': {'train': os.path.join(data_split_root, 'fr-en.train.en'),
                           'valid': os.path.join(data_split_root, 'fr-en.valid.en'),
                           'test': os.path.join(data_split_root, 'fr-en.test.en')}}

cs_de_split = {'czech': {'train': os.path.join(data_split_root, 'cs-de.train.cs'),
                         'valid': os.path.join(data_split_root, 'cs-de.valid.cs'),
                         'test': os.path.join(data_split_root, 'cs-de.test.cs')},
               'german': {'train': os.path.join(data_split_root, 'cs-de.train.de'),
                          'valid': os.path.join(data_split_root, 'cs-de.valid.de'),
                          'test': os.path.join(data_split_root, 'cs-de.test.de')}}

cs_fr_split = {'czech': {'train': os.path.join(data_split_root, 'cs-fr.train.cs'),
                         'valid': os.path.join(data_split_root, 'cs-fr.valid.cs'),
                         'test': os.path.join(data_split_root, 'cs-fr.test.cs')},
               'french': {'train': os.path.join(data_split_root, 'cs-fr.train.fr'),
                          'valid': os.path.join(data_split_root, 'cs-fr.valid.fr'),
                          'test': os.path.join(data_split_root, 'cs-fr.test.fr')}}

de_fr_split = {'german': {'train': os.path.join(data_split_root, 'de-fr.train.de'),
                          'valid': os.path.join(data_split_root, 'de-fr.valid.de'),
                          'test': os.path.join(data_split_root, 'de-fr.test.de')},
               'french': {'train': os.path.join(data_split_root, 'de-fr.train.fr'),
                          'valid': os.path.join(data_split_root, 'de-fr.valid.fr'),
                          'test': os.path.join(data_split_root, 'de-fr.test.fr')}}
