class BaseDataReader:
    def __init__(self):
        super().__init__()

    def load_record(self, data_config, data_reader_name=None, raw_data=None):
        if data_config['loading']['format'] == 'table':
            return self.load_record_from_table(data_config, data_reader_name=data_reader_name)
        elif data_config['loading']['format'] == 'npy':
            return self.load_record_from_npy(data_config, data_reader_name=data_reader_name, raw_data=raw_data)
        elif data_config['loading']['format'] in ['dir', 'directory']:
            return self.load_record_from_dir(data_config, data_reader_name=data_reader_name)
        elif data_config['loading']['format'] == 'pytorch':
            return self.load_record_from_pytorch(data_config, data_reader_name=data_reader_name)
        else:
            raise ValueError(f'Data loading format not supported: {data_config["loading"]["format"]}')

    def load_record_from_dir(self, data_config, data_reader_name=None):
        raise NotImplementedError(f'load_record_from_dir not implemented for {self.__class__.__name__}')

    def load_record_from_table(self, data_config, data_reader_name=None):
        raise NotImplementedError(f'load_record_from_dir not implemented for {self.__class__.__name__}')

    def load_record_from_npy(self, data_config, data_reader_name=None):
        raise NotImplementedError(f'load_record_from_dir not implemented for {self.__class__.__name__}')

    def load_record_from_pytorch(self, data_config, data_reader_name=None):
        raise NotImplementedError(f'load_record_from_dir not implemented for {self.__class__.__name__}')
