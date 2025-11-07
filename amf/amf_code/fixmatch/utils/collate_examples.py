# The point of this script is to collate the training dataset, so that a
# 10% labelled set will always be a strict subset of the 20% labelled set
# and so that as we increase the quantity of unlabelled data, we're just adding
# examples rather than changing them
import argparse
import csv

import numpy as np

# TODO: Unify with original amf scripts to make it work.


def main(args):
    input_dict = dict()
    # Load text file
    with open(args.in_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            try:
                input_dict[row[1]].append(row[0] + "," + str(row[1]))
            except KeyError:
                input_dict[row[1]] = [row[0] + "," + str(row[1])]

    collator = Collator(*list(input_dict.values()))
    collated_list = collator.collate()

    with open(args.out_file, "w") as f:
        for item in collated_list:
            f.write(item + "\n")


class Collator:
    """
    Collates a set of lists, so that any sample taken is proportionally
     representative
    """

    def __init__(self, *args):
        self.example_lists = [example_list for example_list in args]
        self.n_classes = len(self.example_lists)
        self.class_sizes = [len(x) for x in self.example_lists]
        self.total_dataset_size = sum(self.class_sizes)
        self.step_sizes = np.array(self.class_sizes) / self.total_dataset_size
        self.current_step = np.zeros(self.n_classes)
        self.current_step_int = np.zeros(self.n_classes)
        self.collated_list = list()

    def _step(self):
        # Each time we take a step, we check whether any of the lists rounds to
        # the next integer. If it does then we add it to the collated list
        self.current_step += self.step_sizes
        next_step_int = np.round(self.current_step)
        for i, value in enumerate(next_step_int):
            if value > self.current_step_int[i]:
                example = self.example_lists[i].pop()
                self.collated_list.append(example)
        self.current_step_int = next_step_int

    def collate(self):
        while len(self.collated_list) < self.total_dataset_size:
            self._step()

        return self.collated_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str)
    arguments = parser.parse_args()

    main(arguments)
