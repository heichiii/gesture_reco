import torch
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO(r"best.pt")  # load a custom model
    results = model.predict(r"2.png")  # predict on an image
    for result in results:
        img = cv2.imread(result.path)
        img = cv2.resize(img, (640, 640))
        out = f"{result.probs.top1}: {torch.softmax(result.probs.data.cpu(), dim=0)[result.probs.top1]*100.:.3f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 2
        org = (10, 30)
        cv2.putText(img, out, org, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        cv2.imshow("Image", img)
        ch = cv2.waitKey()
        # if ch == ord('s'):
        #     cv2.imwrite(r"C:\Users\ZJZ0\Desktop\python深度学习\datasets/2_res.png", img)
        #     print("save")

    # model.export(format="onnx")