"""
Code given by *Compositionality Decomposed: How do Neural Networks Generalise?*  (Dieuwke Hupkes et al. 2020).
<https://github.com/MathijsMul/pcfg-set/blob/master/generate.py>

Note that we slightly edit the code to check for sample duplicates and allow for data splitting.

Copyright 2020 Mathijs Mul
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random
import sys

from comp_rep.utils import save_list_to_csv


class MarkovTree:
    """
    PCFG
    """

    def __init__(
        self,
        unary_functions,
        binary_functions,
        alphabet,
        prob_unary,
        prob_func,
        lengths,
        placeholders,
        omit_brackets,
    ):
        sys.setrecursionlimit(50)

        self.unary_functions = unary_functions
        self.binary_functions = binary_functions
        self.all_functions = self.unary_functions + self.binary_functions

        self.alphabet = alphabet

        self.set_probabilities(prob_unary, prob_func)

        self.lengths = lengths
        self.string_arguments = []
        self.arg_length_counts = {i: len(self.alphabet) ** i for i in self.lengths}

        self.placeholders = placeholders
        self.omit_brackets = omit_brackets

    def set_probabilities(self, prob_unary, prob_func):
        self.prob_unary = prob_unary
        self.prob_binary = 1 - self.prob_unary

        self.prob_func = prob_func
        self.prob_str = 1 - self.prob_func

    def function_next(self):
        # Determine item following arbitrary function call
        if random.random() < self.prob_unary:
            return [random.choice(self.unary_functions), self.unary_next()]
        else:
            [arg1, arg2] = self.binary_next()
            return [random.choice(self.binary_functions), arg1, arg2]

    def unary_next(self):
        # Determine item following unary function call
        if random.random() < self.prob_func:
            return self.function_next()
        else:
            return self.string_argument()

    def binary_next(self):
        # Determine items following binary function call
        next = []

        for i in range(2):
            if random.random() < self.prob_str:
                next += [self.string_argument()]
            else:
                next += [self.function_next()]
        return next

    def string_argument(self):
        if not self.placeholders:
            candidate = [
                random.choice(self.alphabet) for i in range(random.choice(self.lengths))
            ]
            if candidate not in self.string_arguments:
                # make sure that the same string arguments do not occur
                self.string_arguments += [candidate]
                return candidate
            else:
                return self.string_argument()
        else:
            candidate_len = random.choice(self.lengths)
            if not self.arg_length_counts[candidate_len] == 0:
                self.arg_length_counts[candidate_len] -= 1
                return ["X" for i in range(candidate_len)]
            else:
                return self.string_argument()

    def build(self):
        # Always start with function call
        tree = self.function_next()
        return tree

    def generate_data(self, nr_samples):
        data = [self.build() for i in range(nr_samples)]
        return data

    def evaluate_tree(self, tree):
        # Evaluate output
        if all(isinstance(item, str) for item in tree):
            return tree
        if tree[0] in self.unary_functions:
            return tree[0](self.evaluate_tree(tree[1]))
        elif tree[0] in self.binary_functions:
            return tree[0](self.evaluate_tree(tree[1]), self.evaluate_tree(tree[2]))

    def write(self, tree):
        # Convert tree to string for data file
        if all(isinstance(item, str) for item in tree):
            return " ".join(tree)
        if tree[0] in self.unary_functions:
            if self.omit_brackets:
                return tree[0].__name__ + " " + self.write(tree[1])
            else:
                return (tree[0].__name__ + " ( " + self.write(tree[1])) + " )"
        elif tree[0] in self.binary_functions:
            if self.omit_brackets:
                return (
                    tree[0].__name__
                    + " "
                    + self.write(tree[1])
                    + " , "
                    + self.write(tree[2])
                )
            else:
                return (
                    tree[0].__name__
                    + " ( "
                    + self.write(tree[1])
                    + " , "
                    + self.write(tree[2])
                    + " )"
                )


def generate_data(pcfg_tree, total_samples, file_dir, random_probs, train_ratio=0.8):
    """
    Generate function data. Note that this function has been edited compared to the original code source.
    """
    sample_list = []
    t = pcfg_tree

    for _ in range(total_samples):
        if random_probs:
            t.set_probabilities(prob_unary=random.random(), prob_func=random.random())
        try:
            tree = t.build()
            written_tree = t.write(tree)

            # Control maximum tree size
            if len(written_tree) < 500:
                sample_list.append(written_tree + ";" + " ".join(t.evaluate_tree(tree)))
        except RecursionError:
            pass

    assert len(sample_list) == len(
        set(sample_list)
    ), f"There are duplicates in the data sample! Length of samples: {len(sample_list)}. Length of set of sample: {len(set(sample_list))}"

    train_samples = sample_list[: int(train_ratio * len(sample_list))]
    save_list_to_csv(
        file_path=file_dir / "train.csv",
        data=train_samples,
    )

    test_samples = sample_list[int(train_ratio * len(sample_list)) :]
    save_list_to_csv(
        file_path=file_dir / "test.csv",
        data=test_samples,
    )
