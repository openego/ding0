import os

# LOG_FILE_PATH = 'pickle_log'
# LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')

def pickle_export_logger(log_file_path):
    """
    Creates a list for pickle files that are missing for some reason.
    Most likely the file does not exists @ the pickle file path dir.

    :param log_file_path:
    :return:
    """
    # does the file exist?
    if not os.path.isfile(log_file_path):
        print('ding0 log-file {file} not found. '
              'This might be the first run of the tool. '
              .format(file=log_file_path))
        base_path = os.path.split(log_file_path)[0]
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
            print('The directory {path} was created.'.format(path=base_path))

        with open(log_file_path, 'a') as log:
            log.write("List of missing grid districts:")
            pass
