#! /usr/bin/env/python

"""
Adds the ability to 'include' files in markdown templates. Just main README.md
for now.
"""

import os
import re

file_pairs = [
    # Source -------------------- Destination
    ["src/readme-template.md", "../README.md"],
]

include_syntax = r'\{\{\'(.+)\'\}\}'

for source, destination in file_pairs:
    source_path = os.path.abspath(os.path.dirname(source))
    # Read the template
    with open(source) as f:
        template_contents = f.read()
        m = re.search(include_syntax, template_contents)
        include_path = m.group(1)
        include_path = os.path.join(source_path, include_path)
    # Read the included file
    with open(include_path) as f:
        include_contents = f.read()
    # Write the combined file
    with open(destination, "w+") as f:
        output = re.sub(include_syntax, include_contents, template_contents)
        f.write(output)
