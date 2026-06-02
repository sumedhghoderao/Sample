import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABEL_PATH = "coco_labels.txt"

# Load labels
labels = {}
with open(LABEL_PATH, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1]

# Load Coral model
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[
        tflite.load_delegate(
            "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1"
        )
    ]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]["shape"][1]
input_w = input_details[0]["shape"][2]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    H, W = frame.shape[:2]

    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    boxes = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    classes = interpreter.get_tensor(
        output_details[1]["index"]
    )[0]

    scores = interpreter.get_tensor(
        output_details[2]["index"]
    )[0]

    count = int(
        interpreter.get_tensor(
            output_details[3]["index"]
        )[0]
    )

    for i in range(count):

        if scores[i] < 0.5:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        xmin = int(xmin * W)
        xmax = int(xmax * W)
        ymin = int(ymin * H)
        ymax = int(ymax * H)

        class_id = int(classes[i])

        label = labels.get(class_id, str(class_id))

        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label}: {scores[i]:.2f}",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Coral Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()