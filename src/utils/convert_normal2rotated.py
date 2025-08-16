"""
Utility to convert a dataset of axis‑aligned bounding boxes and keypoints stored
in Pascal VOC XML files into the rotated OBB format expected by the OBB pose
model.  For each object in each XML file, a small random rotation angle is
sampled and applied to both the bounding box and the keypoint, generating
slightly rotated training examples.  The rotated boxes are output as four
corner coordinates (x1,y1,x2,y2,x3,y3,x4,y4) along with the rotated keypoint
(kx,ky).  All coordinates are normalised to the image width and height and
written to a single‐line `.txt` file for each image.

The output format per line is:

    class_id x1 y1 x2 y2 x3 y3 x4 y4 kx ky

Where ``class_id`` is an integer label for the object class and the
coordinates are floats between 0 and 1.  The order of the four points is
clockwise starting from the top‑left corner of the rotated rectangle.

Usage:
    python convert_dataset.py --input_dir path/to/xmls --output_dir path/to/labels

You can specify the rotation range with --angle_min and --angle_max.  By
default the angle is chosen uniformly between 1 and 10 degrees with a
random sign (positive or negative).
"""

import os
import argparse
import math
import xml.etree.ElementTree as ET
import random
from collections import defaultdict

def obb_to_polygon(cx, cy, w, h, angle_rad):
    """Return the four corner coordinates of a rotated rectangle.

    The rectangle is defined by its centre (cx, cy), width w, height h and
    rotation angle in radians.  The corners are returned in clockwise order
    starting from the top‑left corner.
    """
    # axis‑aligned corners around the origin (top‑left, top‑right, bottom‑right, bottom‑left)
    hw, hh = w * 0.5, h * 0.5
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    pts = []
    for x, y in corners:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        pts.append((xr + cx, yr + cy))
    return pts


def rotate_point(x, y, cx, cy, angle_rad):
    """Rotate a point (x, y) around centre (cx, cy) by angle_rad radians."""
    dx, dy = x - cx, y - cy
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    xr = dx * cos_a - dy * sin_a
    yr = dx * sin_a + dy * cos_a
    return xr + cx, yr + cy


def parse_xml(file_path):
    """Parse a Pascal VOC XML file and return image info and object annotations.

    Returns:
        width, height: image dimensions
        objects: list of dicts with keys: 'name', 'bbox' (xmin,xmin,xmax,ymax), 'kpt' (x,y)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        kpt_elem = obj.find('keypoint')
        if kpt_elem is not None:
            kx = float(kpt_elem.find('x').text)
            ky = float(kpt_elem.find('y').text)
        else:
            # default keypoint at centre if missing
            kx = (xmin + xmax) / 2.0
            ky = (ymin + ymax) / 2.0
        objects.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax),
            'kpt': (kx, ky)
        })
    return width, height, objects


def convert_dataset(input_dir, output_dir, angle_min=1.0, angle_max=10.0, random_sign=True):
    """Convert all XML files in input_dir to rotated label files in output_dir.

    Args:
        input_dir: directory containing Pascal VOC XML annotations.
        output_dir: directory to write YOLO OBB label files (.txt).
        angle_min, angle_max: range of rotation angles in degrees.
        random_sign: if True, randomly choose positive or negative sign for each angle.
    """
    os.makedirs(output_dir, exist_ok=True)
    # build a class mapping (assign an integer id to each class name)
    class_map = defaultdict(lambda: None)
    class_counter = 0
    # first pass to collect class names
    xml_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xml')]
    for xml_name in xml_files:
        width, height, objects = parse_xml(os.path.join(input_dir, xml_name))
        for obj in objects:
            name = obj['name']
            if class_map[name] is None:
                class_map[name] = class_counter
                class_counter += 1
    # second pass: convert and write labels
    for xml_name in xml_files:
        xml_path = os.path.join(input_dir, xml_name)
        width, height, objects = parse_xml(xml_path)
        # derive output label path (same stem, .txt extension)
        stem = os.path.splitext(xml_name)[0]
        out_path = os.path.join(output_dir, f"{stem}.txt")
        lines = []
        for obj in objects:
            name = obj['name']
            class_id = class_map[name]
            xmin, ymin, xmax, ymax = obj['bbox']
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = (xmax - xmin)
            h = (ymax - ymin)
            # sample angle in radians
            angle_deg = random.uniform(angle_min, angle_max)
            if random_sign and random.random() < 0.5:
                angle_deg *= -1.0
            angle_rad = math.radians(angle_deg)
            # compute rotated corners
            corners = obb_to_polygon(cx, cy, w, h, angle_rad)
            # rotate keypoint
            kx, ky = obj['kpt']
            kx_rot, ky_rot = rotate_point(kx, ky, cx, cy, angle_rad)
            # normalise coordinates
            coords = []
            for (x, y) in corners:
                coords.append(x / width)
                coords.append(y / height)
            kx_norm = kx_rot / width
            ky_norm = ky_rot / height
            # build line
            values = [class_id] + [float(f"{c:.6f}") for c in coords] + [float(f"{kx_norm:.6f}"), float(f"{ky_norm:.6f}")]
            line = ' '.join(str(v) for v in values)
            lines.append(line)
        # write file
        with open(out_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    print(f"Converted {len(xml_files)} annotation files. Class mapping: {dict(class_map)}")


def main():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to rotated YOLO OBB format")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with XML annotation files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save YOLO OBB label files')
    parser.add_argument('--angle_min', type=float, default=1.0, help='Minimum rotation angle in degrees')
    parser.add_argument('--angle_max', type=float, default=10.0, help='Maximum rotation angle in degrees')
    parser.add_argument('--no_random_sign', action='store_true', help='Disable random sign for rotation angle (always positive)')
    args = parser.parse_args()
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        random_sign=not args.no_random_sign,
    )


if __name__ == '__main__':
    main()