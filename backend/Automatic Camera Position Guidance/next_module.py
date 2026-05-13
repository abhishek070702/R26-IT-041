import sys
from pathlib import Path
import cv2


def main():
    if len(sys.argv) < 2:
        print("No image path received.")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print("Image not found:", image_path)
        return

    print("Next module received image:")
    print(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return

    cv2.imshow("Image Received by Next Module", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()