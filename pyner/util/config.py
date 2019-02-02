import yaml


class ConfigParser:
    def __init__(self):
        self.__name__ = 'config'

    @classmethod
    def parse(cls, config_path):
        with open(config_path, encoding='utf-8') as config_file:
            config_dictionary = yaml.load(config_file)

        key2dictionary = {}
        for key, config_path in config_dictionary.items():
            if not config_path.endswith('.yaml'):
                key2dictionary[key] = config_path

            else:
                config = open(config_path, encoding='utf-8')
                key2dictionary[key] = yaml.load(config)
                config.close()

        configs = cls()
        configs.config_dict = key2dictionary
        return configs

    def __call__(self):
        return self.config_dict.keys()

    def export(self):
        return self.config_dict

    def __getitem__(self, key):
        return self.config_dict[key]

    def __contains__(self, key):
        return key in self.config_dict
