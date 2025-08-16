"""
Simple GUI to visualize rotated bounding boxes and keypoints on images.

Given a directory of images and a corresponding directory of YOLO‑style
OBB labels (created by ``convert_dataset.py``), this script iterates
through the images and displays each one with its rotated bounding boxes
and keypoints drawn on top.  Use the ``n`` key to advance to the next
image, the ``p`` key to go back, and ``q`` or the window close button
to exit.  The class labels are shown in the window title along with
the current image index.

Usage:
    python visualize_labels.py --images_dir path/to/images --labels_dir path/to/labels

The label files must be named identically to the images but with a
``.txt`` extension.  Each line of the label file must contain:

    class_id x1 y1 x2 y2 x3 y3 x4 y4 kx ky

where coordinates are normalized (0–1).  The script automatically
rescales them to pixel coordinates based on the image dimensions.
"""

import argparse
import os
import glob
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw


def parse_label_line(line):
    """Parse a single line of a label file into class id, points and keypoint."""
    parts = line.strip().split()
    if len(parts) != 11:
        raise ValueError(f"Label line must have 11 values, got {len(parts)}: {line}")
    class_id = int(float(parts[0]))
    coords = list(map(float, parts[1:9]))  # x1 y1 ... x4 y4 (normalized)
    kx_norm = float(parts[9])
    ky_norm = float(parts[10])
    # convert to list of point tuples
    pts_norm = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    return class_id, pts_norm, (kx_norm, ky_norm)


class LabelViewer:
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        # gather images (support common extensions)
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths = sorted(self.image_paths)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir}")
        self.index = 0
        # tkinter setup
        self.root = tk.Tk()
        self.root.title("Label Viewer")
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # bind keys
        self.root.bind('<n>', self.next_image)
        self.root.bind('<p>', self.prev_image)
        self.root.bind('<q>', self.quit)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        # load first image
        self.show_image()
        self.root.mainloop()

    def load_label(self, image_path):
        """Load corresponding label file for an image."""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            return []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    labels.append(parse_label_line(line))
                except Exception as e:
                    print(f"Failed to parse line in {label_path}: {e}")
        return labels

    def show_image(self):
        image_path = self.image_paths[self.index]
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        # draw labels
        draw = ImageDraw.Draw(img)
        labels = self.load_label(image_path)
        title_parts = [f"{os.path.basename(image_path)} ({self.index+1}/{len(self.image_paths)})"]
        for class_id, pts_norm, (kx_norm, ky_norm) in labels:
            # convert normalized coordinates to pixel coordinates
            pts = [(x * width, y * height) for (x, y) in pts_norm]
            kx = kx_norm * width
            ky = ky_norm * height
            # draw polygon (closed)
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
            # draw keypoint
            r = 3
            draw.ellipse((kx - r, ky - r, kx + r, ky + r), fill=(0, 255, 0))
            title_parts.append(f"class {class_id}")
        # update window title
        self.root.title(" | ".join(title_parts))
        # display on canvas
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=width, height=height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.show_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.show_image()

    def quit(self, event=None):
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Visualize rotated OBB labels on images")
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory containing label (.txt) files')
    args = parser.parse_args()
    LabelViewer(args.images_dir, args.labels_dir)


if __name__ == '__main__':
    main()