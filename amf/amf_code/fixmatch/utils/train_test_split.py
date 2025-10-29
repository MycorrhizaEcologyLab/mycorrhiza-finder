import argparse
import os
import numpy as np
import random
from fnmatch import fnmatch

# TODO: Unify with original amf scripts to make it work.


def unit_interval_float(arg):
    """ Type function for Argparse - float between 0 and 1"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if not 0 < f < 1:
        raise argparse.ArgumentTypeError("Argument must be between 0 and 1")
    return f


def main(args):

    splitter = Splitter(args)
    train, val, test = splitter.split()
    for set_name, files in {"train": train, "val":val, "test":test}.items():
        # TODO ajg4: rework for consistent filenames
        filename = os.path.join(args.out_dir, f"{set_name}_split.txt")
        with open(filename, "w") as f:
            for item, label in files:
                f.write(item + "," + str(label) + "\n")
        print(f"Written {set_name} set with {len(files)} images as {filename}")


class Splitter:

    def __init__(self, args):

        # set seed
        random.seed(args.seed)

        self.base_dir = args.base_dir
        self.train_pct = args.train_fraction
        self.magnification = args.magnification
        self.multiclass = args.multiclass

        # Infer val percentage
        self.val_pct = (1 - self.train_pct)/2

    def _get_diagnosis_details(self):

        # TODO: Is this a test class? Is this needed?
        class_table = {
            # benign
            "adenosis": 0,
            "fibroadenoma": 1,
            "phyllodes_tumor": 2,
            "tubular_adenoma": 3,
            # malignant
            "ductal_carcinoma": 4,
            "lobular_carcinoma": 5,
            "mucinous_carcinoma": 6,
            "papillary_carcinoma": 7
        }

        out_dict = dict()

        for diagnosis in ["malignant", "benign"]:

            diagnosis_dir = os.path.join(self.base_dir, diagnosis)
            file_type = "*.png"
            diagnosis_list = list()

            for path, subdirs, files in os.walk(diagnosis_dir):
                for file in files:
                    if fnmatch(file, file_type) and self.magnification in path:
                        # list of lists of [filepath, cancer_type, patientID, diagnosis]
                        if self.multiclass:
                            diagnosis_class = class_table[path.split('/')[-3]]
                        else:
                            diagnosis_class = 1 if diagnosis == "malignant" else 0
                        diagnosis_list.append(
                            {"path": os.path.abspath(os.path.join(path, file)),
                             "cancer_type": path.split('/')[-3],
                             "patient_id": path.split('/')[-2],
                             "diagnosis": diagnosis_class})

            out_dict[diagnosis] = diagnosis_list

        return out_dict


    def split(self):

        diagnosis_dict = self._get_diagnosis_details()

        train_list = list()
        val_list = list()
        test_list = list()

        for diagnosis, diag_info_list in diagnosis_dict.items():

            train_patient_list = list()
            val_patient_list = list()
            test_patient_list = list()

            # Create dicts of patients per cancer type
            cancer_type_dict = dict()
            for item in diag_info_list:
                try:
                    cancer_type_dict[item["cancer_type"]].append(item["patient_id"])
                except KeyError:
                    cancer_type_dict[item["cancer_type"]] = [item["patient_id"]]

            for cancer_type, patient_list in cancer_type_dict.items():

                # Remove duplicates
                patient_list = list(set(patient_list))

                n_patients = len(patient_list)

                # Ensure train, val and test have at least one example

                assert n_patients >= 3, "Not enough patients in class to split"
                train_size = 1
                val_size = 1
                test_size = 1

                n_patients -= 3

                train_size += int(np.ceil(n_patients * self.train_pct))
                val_size += int(np.floor(n_patients * self.val_pct))
                test_size += int(np.floor(n_patients * self.val_pct))

                # The above will sometimes leave a remainder, so allocate rem to train
                n_patients -= (train_size + val_size + test_size - 3)  # Don't double count initial ones
                if n_patients > 0:
                    train_size += n_patients

                random.shuffle(patient_list)

                train_patient_list += patient_list[:train_size]
                val_patient_list += patient_list[train_size: train_size + val_size]
                test_patient_list += patient_list[train_size + val_size:]

            for item in diag_info_list:
                if item["patient_id"] in train_patient_list:
                    train_list.append([item["path"], item["diagnosis"]])
                elif item["patient_id"] in val_patient_list:
                    val_list.append([item["path"], item["diagnosis"]])
                elif item["patient_id"] in test_patient_list:
                    test_list.append([item["path"], item["diagnosis"]])

        return train_list, val_list, test_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True, help="directory with 'benign' and 'malginant' as subdirs" )
    parser.add_argument("--train-fraction", type=float, default=0.6, help="fraction of dataset to be used as train. The rest is split evenly between test and val")
    parser.add_argument("--out-dir", type=str, required=True, help="location to output split files")
    parser.add_argument("--magnification", type=str, default="200X", choices={"40X", "100X", "200X", "400X"}, help="which magnification to produce the split for")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument('--multiclass', dest='multiclass', action='store_true')
    parser.set_defaults(multiclass=False)
    args = parser.parse_args()

    main(args)
