import cv2


def test_camera(index):
    print(f"Testing camera index: {index}")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera {index} not opened")
        return

    print(f"Camera {index} opened successfully")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        cv2.putText(
            frame,
            f"Camera Index: {index}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(f"Camera Test {index}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


print("Camera test")
print("Press Q to close camera window")

camera_index = int(input("Enter camera index 0, 1, or 2: "))
test_camera(camera_index)