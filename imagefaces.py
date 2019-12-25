#!/usr/bin/env python3
# Identify faces, calculate their encoding vectors and store this information in a JSON file.

from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
import json
import progressbar
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import face_recognition

progressbar.streams.wrap_stdout()

argparser = ArgumentParser(description="Locate faces in given images, calculate their encoding and store it in a JSON file.")
argparser.add_argument("--patterns", "-p", nargs="+", default=["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"], help="File name patterns to use if directories are recursed (default: %(default)s).")
argparser.add_argument("--output", "-o", default=".", help="Output path (default: %(default)s)")
argparser.add_argument("--jobs", "-j", type=int, help="Number of worker processes (default: number of CPUs)")
argparser.add_argument("--image-resize", "-s", type=int, default=1024, help="Resize images to maximum width/height given by this parameter (default: %(default)s)")
argparser.add_argument("--no-image", "-n", action="store_false", dest="generate_image", help="Don't generate image with face marks.")
argparser.add_argument("--model", "-m", choices=("hog", "cnn"), default="hog", help="Face location model (default: %(default)s)")
argparser.add_argument("images", nargs="+", type=Path, help="Images or directories to process. Recurses into directories.")
args = argparser.parse_args()

paths = []
for path in args.images:
    if path.is_dir():
        paths.extend([
                filepath
                for pattern in args.patterns
                for filepath in path.glob(f"**/{pattern}")
            ])
    else:
        paths.append(path)

def get_exif(img):
    return {
        ExifTags.TAGS.setdefault(k, k): v
        for k,v in dict(img.getexif()).items()
    }

def prepare_image(path):
    img = Image.open(path)
    
    # Downscale to desired size keeping the aspect ratio
    s = args.image_resize
    img.thumbnail((s, s), Image.LINEAR)
    
    # Rotate according to orientation stored in EXIF info
    o = get_exif(img).setdefault("Orientation", 1)
    if o == 8:
        img = img.transpose(Image.ROTATE_90)
    elif o == 3:
        img = img.transpose(Image.ROTATE_180)
    elif o == 6:
        img = img.transpose(Image.ROTATE_270)
    
    return img

def process_image(path):
    json_out_path = f"{args.output}/{path.stem}.json"
    img_out_path = f"{args.output}/{path.name}"
    if not Path(json_out_path).exists():
        img = prepare_image(path)
        imgarray = np.array(img)
        locations = face_recognition.face_locations(imgarray, model=args.model)
        if len(locations) > 0:
            encodings = face_recognition.face_encodings(imgarray, locations)
        else:
            encodings = []

        with open(json_out_path, "w") as f:
            json.dump(
                {
                    "locations": locations,
                    "encodings": [ encoding.tolist() for encoding in encodings ],
                    "source": str(path),
                    "destination": img_out_path,
                },
                f,
                indent=2
            )
        if args.generate_image:
            draw = ImageDraw.Draw(img)
            for location in locations:
                t, r, b, l = location
                draw.rectangle((l, t, r, b), outline="red", width=3)
            img.save(img_out_path)
        return len(locations)
    else:
        return 0

with Pool(processes=args.jobs) as p:
    facecounts = list(progressbar.progressbar(p.imap(process_image, paths), max_value=len(paths)))
print(f"Identified {sum(facecounts)} faces in {len(paths)} images.")
