"""
Code modified based on *Compositionality Decomposed: How do Neural Networks Generalise?*  (Dieuwke Hupkes et al. 2020).
<https://github.com/MathijsMul/pcfg-set/blob/master/tasks/default.py

Copyright 2020 Mathijs Mul
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


# UNARY
def copy(sequence):
    return sequence


def reverse(sequence):
    return sequence[::-1]


def shift(sequence):
    return sequence[1:] + [sequence[0]]


def echo(sequence):
    return sequence + [sequence[-1]]


def swap_first_last(sequence):
    return [sequence[-1]] + sequence[1:-1] + [sequence[0]]


def repeat(sequence):
    return sequence + sequence


# BINARY
def append(sequence1, sequence2):
    return sequence1 + sequence2


def prepend(sequence1, sequence2):
    return sequence2 + sequence1


def remove_first(sequence1, sequence2):
    return sequence2


def remove_second(sequence1, sequence2):
    return sequence1


UNARY_FUNC = ["copy", "reverse", "shift", "echo", "swap_first_last", "repeat"]
BINARY_FUNC = ["append", "prepend", "remove_first", "remove_second"]

FUNC_MAP = {
    "copy": copy,
    "reverse": reverse,
    "shift": shift,
    "echo": echo,
    "swap_first_last": swap_first_last,
    "repeat": repeat,
    "append": append,
    "prepend": prepend,
    "remove_first": remove_first,
    "remove_second": remove_second,
}
