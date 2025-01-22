class BaseDatum:
    def __init__(self, data_dict):        
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, key):
        return self.data_dict[key]
    
    def __setitem__(self, key, newvalue):
        self.data_dict[key] = newvalue
    
    def keys(self):
        return self.data_dict.keys()
    def get(self, key, default=None):
        return self.data_dict.get(key, default)
    
    def get_data(self):
        return self.data_dict

    def feed_to_network(self):
        return {key:self.data_dict[key] for key in self.data_dict['feed_to_network_roles']}
    
    def load_datum(self):
        pass

    def update(self, new_dict):
        self.data_dict.update(new_dict)

    def items(self):
        return self.data_dict.items()
    
    def values(self):
        return self.data_dict.values()
    
    def __repr__(self):
        return self.data_dict.__repr__()
    
    def __str__(self):
        return self.data_dict.__str__()
    
    

    @staticmethod
    def load_data(data_list):
        pass

if __name__ == '__main__':
    import numpy as np
    datum = BaseDatum({'image': np.zeros((5,5))})
    datum['image'] = datum['image'] + 1
    print(datum['image'])


