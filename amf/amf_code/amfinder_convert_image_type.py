import gc
import os
import xml.etree.ElementTree as ET

from PIL import Image

import amfinder_config as AmfConfig
import amfinder_log as AmfLog

# This allows any size image
Image.MAX_IMAGE_PIXELS = None


def convert_tiff(path, file_type="jpg"):
    if (
        os.path.splitext(path)[1].lower() == ".tiff"
        or os.path.splitext(path)[1].lower() == ".tif"
    ):
        # You MUST export the annotations along with the TIFF by changing the slidemaster settings
        # This changes the image offset - if you export the annotations separately this does not work
        try:
            im = Image.open(path)
            AmfLog.info(f"Original image size: {im.size}")

            # Check if annotation file exists - if not, convert the entire tif
            if os.path.isfile(os.path.splitext(path)[0] + ".xml"):
                AmfLog.info(
                    f"Found annotations in {os.path.splitext(path)[0]}.xml, cropping tif file to match the given annotations"
                )
                # Parse annotations from XML provided by slidemaster - this needs
                # to have the exact same name as the TIF image you want to parse
                tree = ET.parse(os.path.splitext(path)[0] + ".xml")
                root = tree.getroot()
                annotations = root.findall(".//annotation")

                annotation_coords = {}

                # Loop through the annotations and get the bounding boxes
                for annotation in annotations:
                    name = annotation.attrib["name"]

                    p_tags = annotation.findall(".//p")
                    min_x = float("inf")
                    min_y = float("inf")
                    max_x = -float("inf")
                    max_y = -float("inf")

                    for p in p_tags:
                        attribs = p.attrib
                        if int(attribs["x"]) < min_x:
                            min_x = int(attribs["x"])

                        if int(attribs["y"]) < min_y:
                            min_y = int(attribs["y"])

                        if int(attribs["x"]) > max_x:
                            max_x = int(attribs["x"])

                        if int(attribs["y"]) > max_y:
                            max_y = int(attribs["y"])

                    print(
                        f"Annotation {name} has top left coord {(min_x, min_y)} and bottom right coord {(max_x, max_y)}"
                    )

                    annotation_coords[name] = [
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                    ]

                for image, coords in annotation_coords.items():
                    min_x, min_y, max_x, max_y = coords
                    # Define crop box for each annotation
                    crop_box = (
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                    )

                    try:
                        # Validate crop box size
                        box_width = crop_box[2] - crop_box[0]
                        box_height = crop_box[3] - crop_box[1]

                        # Proceed if valid
                        if box_width > 0 and box_height > 0:
                            print(f"{image} has crop box {crop_box}")
                            # Crop image to box
                            crop = im.crop(crop_box)
                            outfile_crop = (
                                f"{os.path.splitext(path)[0]}_{image}.{file_type}"
                            )
                            if os.path.isfile(outfile_crop):
                                print(
                                    f"JPEG already exists: {outfile_crop}. Skipping..."
                                )
                                continue
                            # Values set to highest possible quality as per docs
                            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
                            crop.convert("RGB").save(
                                outfile_crop, subsampling=0, quality=95
                            )
                            print(f"Saving image to {outfile_crop}")
                        else:
                            print(f"Invalid crop box for image {image}: {crop_box}")
                    except Exception as e:
                        print(
                            f"Failed to crop {image} due to error: {e.__class__.__name__}: {e}"
                        )
            else:
                AmfLog.info(
                    f"No annotations found for {os.path.splitext(path)[0]}, directly converting TIF to {file_type}"
                )
                outfile_crop = f"{os.path.splitext(path)[0]}.{file_type}"
                im.convert("RGB").save(outfile_crop, subsampling=0, quality=95)

            # Close image to free memory
            im.close()
            gc.collect()
        except Exception as e:
            im.close()
            gc.collect()
            print(f"Failed to crop due to error: {e.__class__.__name__}: {e}")


def run(input_images):
    file_type = AmfConfig.get("convert_image_file_type")
    AmfLog.info(f"Running conversion of tifs to {file_type}s")

    for path in input_images:
        convert_tiff(path, file_type)

    return 200
