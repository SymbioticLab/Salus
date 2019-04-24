# -*- coding: future_fstrings -*-
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
# 
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import absolute_import, print_function, division
from builtins import input


def choose(prompt, choices=None, default=0):
    """Prompt the user to make a choice.

    Args:
        prompt: The prompt to show

        choices: Iterable of tuples of choices. Each tuple represents a choice, and is
        in the form (one letter, help, value) or (one letter, help). If value is missing,
        it defaults to the letter.
        The default choices are [('y', 'yes', True), ('n', 'no', False)]

        default: the index of the default choice. Defaults to 0

    Returns:
        the associated value of the choice the user has made.
    """
    # Handle default arguments
    if choices is None:
        choices = [('y', 'yes', True), ('n', 'no', False)]

    # validate arguments
    if not choices:
        raise ValueError('Empty choices')
    if default < 0 or default >= len(choices):
        raise IndexError(f'Default index should be within [0, {len(choices)}), got: {default}')

    def parse_choice(ch):
        if len(ch) == 2:
            return ch[0].lower(), ch[1], ch[0]
        elif len(ch) == 3:
            return ch[0].lower(), ch[1], ch[2]
        else:
            raise ValueError(f'Invalid choice in choices: {tuple}')

    choices = [parse_choice(c) for c in choices]

    # form choices string
    choices_str = '/'.join(ch[0] if idx != default else ch[0].upper()
                           for idx, ch in enumerate(choices))

    prompt = f'{prompt} [{choices_str}]: '
    def_resp = choices[default][0]
    while True:
        resp = input(prompt)
        if not resp:
            resp = def_resp
        resp = resp.lower()

        for ch, _, value in choices:
            if resp == ch:
                return value

        # Invalid input, print help
        print(f'Invalid response: {resp}')
        print('Accepted responses are:')
        for ch, h, _ in choices:
            print(f'{ch} - {h}')


def confirm(prompt, default=False, yes_choice='y', no_choice='n'):
    """Prompt for user's confirmation on some operation.

    Returns:
        True if the user confirmed, False otherwise.
    """
    return choose(prompt, choices=[(yes_choice, 'yes', True), (no_choice, 'no', False)], default=0 if default else 1)


def pause(prompt='Press enter to continue...'):
    """Pause the execution and wait the user to press enter"""

    # we don't want to guard against KeyboardInterrupt
    try:
        input(prompt)
    except EOFError:
        pass
