import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import requests
import time

# --------------------------------------------------
# PATHS
# --------------------------------------------------

CORAL_MODEL = "/home/khadas/Documents/coral_test/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"

LABELS_FILE = "/home/khadas/Documents/coral_test/coco_labels.txt"

# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

labels = {}

with open(LABELS_FILE, "r") as f:
    for i, line in enumerate(f):
        labels[i] = line.strip()

# --------------------------------------------------
# LOAD CORAL MODEL
# --------------------------------------------------

interpreter = tflite.Interpreter(
    model_path=CORAL_MODEL,
    experimental_delegates=[
        tflite.load_delegate(
            "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1"
        )
    ]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_H = input_details[0]["shape"][1]
INPUT_W = input_details[0]["shape"][2]

# --------------------------------------------------
# QWEN SERVER
# --------------------------------------------------

def ask_qwen(relation):

    prompt = f"""
Relationship:
{relation}

Describe this naturally in one short sentence.
"""

    try:

        response = requests.post(
            "http://127.0.0.1:8080/completion",
            json={
                "prompt": prompt,
                "n_predict": 30,
                "temperature": 0.2
            },
            timeout=10
        )

        data = response.json()

        return data.get("content", "")

    except Exception as e:

        return f"Qwen Error: {e}"

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

last_relation = ""
detection_start_time = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    H, W = frame.shape[:2]

    img = cv2.resize(frame, (INPUT_W, INPUT_H))
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

    bottle = None
    mouse = None

    for i in range(count):

        if scores[i] < 0.5:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        xmin = int(xmin * W)
        xmax = int(xmax * W)

        ymin = int(ymin * H)
        ymax = int(ymax * H)

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        class_id = int(classes[i])

        label = labels.get(class_id, str(class_id))

        # ---------------------------------
        # ONLY BOTTLE + MOUSE
        # ---------------------------------

        if label not in ["bottle", "mouse"]:
            continue

        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if label == "bottle" and bottle is None:

            bottle = {
                "cx": cx,
                "cy": cy
            }

        elif label == "mouse" and mouse is None:

            mouse = {
                "cx": cx,
                "cy": cy
            }

    # ---------------------------------
    # RELATIONSHIP ENGINE
    # ---------------------------------

    if bottle is not None and mouse is not None:

        if detection_start_time is None:

            detection_start_time = time.time()

        elapsed = time.time() - detection_start_time

        cv2.putText(
            frame,
            f"Stable Detection: {elapsed:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if elapsed >= 5:

            dx = abs(
                bottle["cx"] - mouse["cx"]
            )

            if bottle["cx"] < mouse["cx"]:

                relation = (
                    "The bottle is to the left "
                    "of the mouse"
                )

            else:

                relation = (
                    "The bottle is to the right "
                    "of the mouse"
                )

            if dx < 120:

                relation += (
                    " and very close to it"
                )

            cv2.putText(
                frame,
                relation,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if relation != last_relation:

                print(
                    "\n=============================="
                )

                print("RELATION:")
                print(relation)

                answer = ask_qwen(
                    relation
                )

                print("\nQWEN:")
                print(answer)

                last_relation = relation

    else:

        detection_start_time = None

    cv2.imshow(
        "Coral + Qwen Demo",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
