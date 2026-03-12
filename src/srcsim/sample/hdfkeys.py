import tables


def has_key(file_name, key):
    with tables.open_file(file_name) as table:
        has_key = key in table

    return has_key


def choose_first_valid_key(file_name, keys):
    for key in keys:
        if has_key(file_name, key):
            return key

    return None


def get_events_key(file_name):
    keys = (
        '/events/parameters',
        '/dl1/event/telescope/parameters/LST_LSTCam',
        '/dl2/event/telescope/parameters/LST_LSTCam'
    )
    key = choose_first_valid_key(file_name, keys)
    return key


def get_config_key(file_name):
    keys = (
        '/simulation/config',
        '/simulation/run_config'
    )
    key = choose_first_valid_key(file_name, keys)
    return key
