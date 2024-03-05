#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np
import cv2

import labelme
from p_tqdm import p_map


def remove_zero_size_bbox(mask):
    small_mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    unique_ids = np.unique(small_mask)
    unique_org_ids = np.unique(mask)
    for u_id in unique_ids:
        curr_mask = small_mask == u_id
        y, x = np.where(curr_mask != 0)
        if (abs(max(y) - min(y)) <= 1 or abs(max(x) - min(x)) <= 1):
            mask[mask == u_id] = 0

    for u_id in unique_org_ids:
        if (u_id not in unique_ids):
            mask[mask == u_id] = 0

    return mask


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input annotated directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument(
        "--labels", help="Labels file or comma separated text", required=True
    )
    parser.add_argument(
        "--noobject", help="Flag not to generate object label", action="store_true"
    )
    parser.add_argument(
        "--nonpy", help="Flag not to generate .npy files", action="store_true"
    )
    parser.add_argument(
        "--noviz", help="Flag to disable visualization", action="store_true"
    )
    parser.add_argument(
        "--no_save_cls_ins_images", help="Flag to disable image saving for class and instance", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.no_save_cls_ins_images:
        os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.nonpy:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassNpy"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassVisualization"))
    if not args.noobject:
        if not args.no_save_cls_ins_images:
            os.makedirs(osp.join(args.output_dir, "SegmentationObject"))
        if not args.nonpy:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectNpy"))
        if not args.noviz:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectVisualization"))
    print("Creating dataset:", args.output_dir)

    if osp.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]

    class_names = []
    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = label.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    def convert_file(filename):
        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_clsp_file = osp.join(args.output_dir, "SegmentationClass", base + ".png")
        if not args.nonpy:
            out_cls_file = osp.join(
                args.output_dir, "SegmentationClassNpy", base + ".npy"
            )
        if not args.noviz:
            out_clsv_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        if not args.noobject:
            out_insp_file = osp.join(
                args.output_dir, "SegmentationObject", base + ".png"
            )
            if not args.nonpy:
                out_ins_file = osp.join(
                    args.output_dir, "SegmentationObjectNpy", base + ".npy"
                )
            if not args.noviz:
                out_insv_file = osp.join(
                    args.output_dir,
                    "SegmentationObjectVisualization",
                    base + ".jpg",
                )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.
        ins = remove_zero_size_bbox(ins)

        # class label
        if not args.no_save_cls_ins_images:
            labelme.utils.lblsave(out_clsp_file, cls)
        if not args.nonpy:
            np.savez_compressed(out_cls_file, cls=cls)
        if not args.noviz:
            clsv = imgviz.label2rgb(
                cls,
                imgviz.rgb2gray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        if not args.noobject:
            # instance label
            if not args.no_save_cls_ins_images:
                labelme.utils.lblsave(out_insp_file, ins)
            if not args.nonpy:
                np.savez_compressed(out_ins_file, ins=ins)
            if not args.noviz:
                instance_ids = np.unique(ins)
                instance_names = [str(i) for i in range(max(instance_ids) + 1)]
                insv = imgviz.label2rgb(
                    ins,
                    imgviz.rgb2gray(img),
                    label_names=instance_names,
                    font_size=15,
                    loc="rb",
                )
                imgviz.io.imsave(out_insv_file, insv)

    list_of_files = sorted(glob.glob(osp.join(args.input_dir, "*.json")))

    results = p_map(convert_file, list_of_files, num_cpus=10)


if __name__ == "__main__":
    main()
