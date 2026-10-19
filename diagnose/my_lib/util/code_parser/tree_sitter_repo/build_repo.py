# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tree_sitter import Language
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('Tree-sitter language library building...')
Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-python',
    'tree-sitter-java',
    # 'tree-sitter-go',
    # 'tree-sitter-javascript',
    # 'tree-sitter-php',
    # 'tree-sitter-ruby',
    # 'tree-sitter-c-sharp',
  ]
)