#!/usr/bin/env python3

import os
import sys
import argparse


def absolute_file_paths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        # f.write(line.rstrip('\r\n') + '\n' + content)
        f.write(line + '\n' + content)


def openego_header():
    """

    Returns
    -------
    header : str
        openego group py-file header
    """

    # define header string
    header = "\"\"\"This file is part of DING0, the DIstribution Network GeneratOr.\n"\
        "DING0 is a tool to generate synthetic medium and low voltage power\n"\
        "distribution grids based on open data.\n" \
        "\n" \
        "It is developed in the project open_eGo: https://openegoproject.wordpress.com\n"\
        "\n" \
        "DING0 lives at github: https://github.com/openego/ding0/\n"\
        "The documentation is available on RTD: http://ding0.readthedocs.io\"\"\"\n"\
        "\n"\
        "__copyright__  = \"Reiner Lemoine Institut gGmbH\"\n"\
        "__license__    = \"GNU Affero General Public License Version 3 (AGPL-3.0)\"\n"\
        "__url__        = \"https://github.com/openego/ding0/blob/master/LICENSE\"\n"\
        "__author__     = \"nesnoj, gplssm\""\
        "\n\n"


    return header #textwrap.dedent(header)

if __name__ == '__main__':
    # setup argparse and parse provided arguments
    parser = argparse.ArgumentParser(
        description="Prepend open_eGo python file header recursively " \
                    "to all files of a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('dir', type=str, help='Directory to walk through')

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    dir_name = args.dir

    # get list of files
    files = absolute_file_paths(dir_name)

    # get header
    header = openego_header()

    # # iterate over file and add header
    for f in files:
        if f.endswith('.py'):
            line_prepender(f, header)