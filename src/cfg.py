
ARGS = {
    "pyro_port": 58116,
    "DB_name": "OpenSet",
    "DB_user": "postgres",
    "DB_host": "127.0.0.1",
    "DB_pw": "password",
    "DB_port": 5432,
    "interface": "any",
    "Dash_port": 8050
}


def get_attribute(att):
    if att in ARGS.keys():
        return ARGS[att]


class config_interface():
    def __init__(self):
        pass

    def __call__(self, att):
        return get_attribute(att)
