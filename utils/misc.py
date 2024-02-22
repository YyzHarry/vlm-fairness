import sys
import torch
import pickle


def pickle_save(filename, obj):
    filename = str(filename)
    if sys.platform == 'darwin':
        # Mac Pickle cannot write file larger than 2GB
        return mac_pickle_dump(filename, obj)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=4)


def pickle_load(filename):
    filename = str(filename)
    if sys.platform == 'darwin':
        # Mac Pickle cannot read file larger than 2GB
        return mac_pickle_load(filename)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def mac_pickle_load(file_path):
    import os
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def mac_pickle_dump(filename, obj):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj, protocol=4)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_json(json_path):
    import json
    assert json_path.exists(), f"{json_path} not found, please create first"
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def save_json(json_path, data):
    import json
    with open(json_path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()
