import argparse
from pathlib import Path

import cv2
import imutils
import numpy as np
from PIL import Image

from models.Net import net_self


LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the custom CNN on sample images.")
    parser.add_argument("--weights", default="log-self/self.h5", help="Path to trained weights.")
    parser.add_argument("--image-dir", default="images", help="Directory of images to predict.")
    parser.add_argument("--show", action="store_true", help="Display images with OpenCV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    image_dir = Path(args.image_dir)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    model = net_self()
    model.build(input_shape=(None, 32, 32, 3))
    model.load_weights(str(weights_path))

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        img = Image.open(image_path).convert("RGB")
        img = img.resize((32, 32))
        img_array = np.array(img).reshape(-1, 32, 32, 3).astype("float32") / 255

        prediction = model.predict(img_array, verbose=0)
        final_prediction = int(np.argmax(prediction))
        probability = float(np.max(prediction))

        print(f"{image_path.name}: {LABEL_NAMES[final_prediction]} ({probability:.4f})")

        if args.show:
            image = cv2.imread(str(image_path))
            image = imutils.resize(image, width=450)
            cv2.imshow("prediction", image)
            cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
