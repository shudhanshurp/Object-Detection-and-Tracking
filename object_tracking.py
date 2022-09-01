from inspect import currentframe
from lib2to3.pgen2.token import DOT
from tkinter import SOLID
from tokenize import Pointfloat
import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import random
from PIL import Image as im
from kalmanfilter import KalmanFilter

od = ObjectDetection()

kf = KalmanFilter()

# tr = EuclideanDistTracker()

cap = cv2.VideoCapture("./test2.mp4")

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter('./video.mp4', fourcc, 1, (1080, 1920))
center_points = {}
prev_first_point = []
first_og_points = []
first_points = {}
id_count = 0
count = 0
center_points_prev_frame = []
center_points_tempprev_frame = []
tracking_objects = {}
track_id = 0
predicted = ''
print("LET'S Start")

while count < 5000:
    ret, frame = cap.read()

    count += 1

    if not ret:
        break
    first_Frame_centerPoint = []
    objects_bbs_ids = []

    object_track_id = []
    center_points_cur_frame = []
    center_points_temp_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    for box in boxes:
        (x, y, w, h) = box
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2

        # cv2.circle(frame, predicted, 3, (255, 255, 255), 2)
        center_points_cur_frame.append((cx, cy))
        center_points_temp_frame.append((x, y, w, h))

        same_object_detected = False
        for id, pt in center_points.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])

            if dist < 25:
                center_points[id] = (cx, cy)
                # print(self.center_points)
                objects_bbs_ids.append((x, y, w, h, id))
                same_object_detected = True
                break

        # New object is detected we assign the ID to that object
        if same_object_detected is False:
            center_points[id_count] = (cx, cy)
            first_points[id_count] = (cx, cy)
            prev_first_point += first_og_points + [[cx, cy, id_count]]
            objects_bbs_ids.append((x, y, w, h, id_count))
            id_count += 1

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 10)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"--------------------------------{prev_first_point}")
    # Only at the beginning we compare previous and current frame

    # object_track_id = tr.update(center_points_temp_frame)
    # print(f"-------------------------------{object_track_id[1][0]}")
    # for i in object_track_id[1]:
    #     print(f"-------------------------------{i}")
    #     first_Frame_centerPoint.append(i)
    # print(fp)
    # print(f"-------------------------------{first_Frame_centerPoint}")
    for pt in center_points_cur_frame:
        for pt2 in center_points_prev_frame:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < 2000:
                tracking_objects[track_id] = pt
                track_id += 1

    for box_id in objects_bbs_ids:
        x, y, w, h, id = box_id
        cxt = int((x + x + w) / 2)
        cyt = int((y + y + h) / 2)

        for firstOG in prev_first_point:
            pointx, pointy, idf = firstOG
            diffx = cxt - pointx
            diffy = cyt - pointy

            cv2.putText(frame, str(id), (x, y - 15),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            if id == idf:
                cv2.line(
                    frame, (pointx, pointy), (cxt, cyt), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

                cv2.line(frame, (cxt, cyt), (diffx+cxt, diffy+cyt), (random.randint(
                    0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # if distance < 10:
        #     cv2.line(frame, pt, pt2, (0, 255, 255), 2)d
        # cv2.line(frame, pt, pt2, (0, 255, 255), 2)

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 3, (0, 0, 255), 2)

        # cv2.line(frame, center_points_cur_frame[2],
        #          center_points_prev_frame[2], (20, 200, 255), 2)

        # cv2.line(frame, point2, pt, (200, 100, 100), 2)
    # print("Tracking objects")
    # print(tracking_objects)

    # print("CUR FRAME LEFT PTS")
    # print(center_points_prev_frame)

    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()
    center_points_tempprev_frame = center_points_temp_frame.copy()
    # prev_first_point = first_og_points.copy()
    # prev_first_point = fp.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

    # else:

    #     tracking_objects_copy = tracking_objects.copy()
    #     center_points_cur_frame_copy = center_points_cur_frame.copy()

    #     for object_id, pt2 in tracking_objects_copy.items():
    #         object_exists = False
    #         for pt in center_points_cur_frame_copy:
    #             distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

    #             # Update IDs position
    #             if distance < 20:
    #                 tracking_objects[object_id] = pt
    #                 object_exists = True
    #                 if pt in center_points_cur_frame:
    #                     center_points_cur_frame.remove(pt)
    #                 continue

    #         # Remove IDs lost
    #         if not object_exists:
    #             tracking_objects.pop(object_id)

    #     # Add new IDs found
    #     for pt in center_points_cur_frame:
    #         tracking_objects[track_id] = pt
    #         track_id += 1

    # data = im.fromarray(frame)
    # data.save(f'./images/{count}.jpg')
    # img = cv2.imread(f'./images/{count}.jpg')
    # video.write(img)
    # Make a copy of the points


# video.release()
cap.release()
cv2.destroyAllWindows()
