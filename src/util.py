import os


def create_dir(*argv):
    dir = os.path.join(*argv)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
