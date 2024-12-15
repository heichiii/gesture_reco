from django.shortcuts import render
import torch
from ultralytics import YOLO
import cv2
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import json
import numpy as np



model = YOLO(r"hand-yolov11n-pose.pt")  # load a custom model

@csrf_exempt
def recognize(request):

    if request.method == 'POST':
        try:
            # 获取请求体并解码为 JSON 格式
            data = json.loads(request.body)
            image_data = data.get('image')
            # 对 base64 编码的图像数据进行解码
            image_bytes = base64.b64decode(image_data)
            img_array = np.fromstring(image_bytes, np.uint8)  # 转换np序列
            img_raw = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 转换Opencv格式BGR

            results = model.predict(img_raw)  # predict on an image

            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                # masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints object for pose outputs
                # probs = result.probs  # Probs object for classification outputs
                # obb = result.obb  # Oriented boxes object for OBB outputs
                # print(keypoints)
                for point in keypoints[0]:
                    # 将浮点数坐标转换为整数
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(img_raw, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

                # result.show()

            # cv2.imshow('image', img_raw)
            # cv2.waitKey(1)
            # response = HttpResponse(img_raw, content_type='image/jpeg')
            # return response
            _, buffer = cv2.imencode('.jpg', img_raw)
            response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
            return response
            #return JsonResponse({"number": number})
        except Exception as e:
            print("Error processing request:", e)
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


def index(request):
    data = {'number': 'aaa'}
    return render(request, 't3.html', data)
