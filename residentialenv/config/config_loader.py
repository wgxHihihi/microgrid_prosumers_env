"""
yaml文件转为属性类
"""
import yaml


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def load_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_config(path):
    mydict = load_yaml(path)
    if not isinstance(mydict, dict):
        print('yaml format error')
        return mydict
    args = []
    for k, v in mydict.items():
        args.append(dictToObj(v))
    return args

# path = '../config/buildings_config.yaml'
# config = load_yaml(path)
# print(config)
# buildingargs = load_buildings_config(config)
# args = dictToObj(config)
# # print(buildingargs[0].del_Tac)
# print(len(buildingargs))
