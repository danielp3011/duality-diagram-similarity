import pickle


def save_dict(filename_, di_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        ret_di = u.load()
        ret_di = pickle.load(f)
        return ret_di

