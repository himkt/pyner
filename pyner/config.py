import yaml


class ConfigParser:
    def __init__(self, config_path):
        with open(config_path, encoding='utf-8') as config_file:
            config_dict = yaml.load(config_file)

        self.config_dict = {}
        for key, config_path in config_dict.items():
            if not config_path.endswith('.yaml'):
                self.config_dict[key] = config_path

            else:
                config = open(config_path, encoding='utf-8')
                self.config_dict[key] = yaml.load(config)
                config.close()

    def __getitem__(self, key):
        return self.config_dict[key]

    def __call__(self):
        return self.config_dict.keys()
