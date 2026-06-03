import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time

# --------------------------------------------------
# PATHS
# --------------------------------------------------

CORAL_MODEL = "/home/khadas/Documents/coral_test/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
QWEN_MODEL = "/home/khadas/Documents/coral_test/qwen2-0_5b-instruct-q4_k_m.gguf"
LABELS_FILE = "/home/khadas/Documents/coral_test/coco_labels.txt"
LLAMA_CLI = "/home/khadas/llama.cpp/build/bin/llama-cli"

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
# QWEN FUNCTION
# --------------------------------------------------

def ask_qwen(scene_text):

    prompt = f"""
You are a vision assistant.

Detected objects and relationships:

{scene_text}

Describe the scene in one short sentence.
"""

    result = subprocess.run(
        [
            LLAMA_CLI,
            "-m",
            QWEN_MODEL,
            "-c",
            "512",
            "-n",
            "50",
            "-p",
            prompt
        ],
        capture_output=True,
        text=True
    )

    return result.stdout

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

last_qwen_time = 0

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

    objects = []

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

        objects.append(
            {
                "label": label,
                "cx": cx,
                "cy": cy
            }
        )

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
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------------------------------
    # RELATIONSHIP ENGINE
    # --------------------------------------------

    relationships = []

    if len(objects) >= 2:

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):

                a = objects[i]
                b = objects[j]

                dx = abs(a["cx"] - b["cx"])

                if a["cx"] < b["cx"]:
                    relation = f"{a['label']} is left of {b['label']}"
                else:
                    relation = f"{a['label']} is right of {b['label']}"

                if dx < 120:
                    relation += " and near it"

                relationships.append(relation)

    # --------------------------------------------
    # QWEN EVERY 5 SECONDS
    # --------------------------------------------

    now = time.time()

    if now - last_qwen_time > 5:

        scene_text = "\n".join(relationships)

        if scene_text:

            print("\n" + "=" * 60)
            print("RELATIONSHIPS")
            print(scene_text)

            response = ask_qwen(scene_text)

            print("\nQWEN:")
            print(response)

        last_qwen_time = now

    cv2.imshow("Coral + Qwen Demo", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
