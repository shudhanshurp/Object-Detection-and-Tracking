from asyncio.windows_events import NULL
from inspect import currentframe
from lib2to3.pgen2.token import DOT
from tkinter import SOLID
from tokenize import Pointfloat
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
from object_detection import ObjectDetection
import math
import random
from PIL import Image as im
from kalmanfilter import KalmanFilter
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from numpy import unique
from numpy import where
from sklearn.neighbors import LocalOutlierFactor


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
lane_points = []
objects_in_lane = [0, 0, 0, 0, 0, 0]
tracking_objects = {}
track_id = 0
predicted = ''
new_array_for_allpoints = []  # store every single points in the video
mid_point_for_lane = {}
print("LET'S Start")

while count < 4000:
    ret, frame = cap.read()

    count += 1
    stable = count * 0.5

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
        new_array_for_allpoints.append((cx, cy))

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
        # print(f"--------------------------------{prev_first_point}")
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

    # vehicle counter code
        if pt[0] > 500 and pt[0] < 600 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[0] += 1
        elif pt[0] > 600 and pt[0] < 700 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[1] += 1
        elif pt[0] > 700 and pt[0] < 800 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[2] += 1
        elif pt[0] > 800 and pt[0] < 900 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[3] += 1
        elif pt[0] > 900 and pt[0] < 1025 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[4] += 1
        elif pt[0] > 1025 and pt[0] < 1150 and pt[1] < 602 and pt[1] > 598:
            objects_in_lane[5] += 1

    for pt in center_points_cur_frame:
        if pt[0] < 500 and pt[0] > 400:
            objects_in_lane.append(pt[0])

    for box_id in objects_bbs_ids:
        x, y, w, h, id = box_id
        cxt = int((x + x + w) / 2)
        cyt = int((y + y + h) / 2)

        for firstOG in prev_first_point:
            pointx, pointy, idf = firstOG
            diffx = cxt - pointx
            diffy = cyt - pointy

        #     cv2.putText(frame, str(id), (x, y - 15),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        #     if id == idf:
        #         cv2.line(
        #             frame, (pointx, pointy), (cxt, cyt), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

        #         cv2.line(frame, (cxt, cyt), (diffx+cxt, diffy+cyt), (random.randint(
        #             0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # if distance < 10:
        #     cv2.line(frame, pt, pt2, (0, 255, 255), 2)
        # cv2.line(frame, pt, pt2, (0, 255, 255), 2)

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 3, (0, 0, 255), 2)

    cv2.line(frame, (525, 600), (625, 600), (0, 255, 255), 2)
    cv2.line(frame, (625, 600), (725, 600), (255, 0, 255), 2)
    cv2.line(frame, (725, 600), (825, 600), (255, 255, 0), 2)
    cv2.line(frame, (825, 600), (925, 600), (120, 120, 255), 2)
    cv2.line(frame, (925, 600), (1050, 600), (120, 255, 255), 2)
    cv2.line(frame, (1050, 600), (1175, 600), (255, 120, 255), 2)

    cv2.putText(frame, "Lane 1:-" +
                str(objects_in_lane[0]), (500, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Lane 2:-" +
                str(objects_in_lane[1]), (600, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Lane 3:-" +
                str(objects_in_lane[2]), (700, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Lane 4:-" +
                str(objects_in_lane[3]), (800, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Lane 5:-" +
                str(objects_in_lane[4]), (900, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Lane 6:-" +
                str(objects_in_lane[5]), (1000, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.line(frame, (1100, 600), (1200, 600), (0, 255, 255), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi /
                            180, threshold=100, minLineLength=5, maxLineGap=250)

    # for line in lines:
    #     print(line)
    #     if line[0] == []:
    #         break
    #     else:
    #         x1, y1, x2, y2 = line[0]
    #     # Filter out the lines in the top of the image
    #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # cv2.line(frame, center_points_cur_frame[2],
    #          center_points_prev_frame[2], (20, 200, 255), 2)

    # cv2.line(frame, point2, pt, (200, 100, 100), 2)
    # print("Tracking objects")
    # print(tracking_objects)

    # print("CUR FRAME LEFT PTS")
    # print(center_points_prev_frame)

    db = DBSCAN(eps=20, min_samples=20)
    db.fit(new_array_for_allpoints)
    # cluster_centers = db.components_

    labels = db.labels_
    numCluster = len(np.unique(labels))
    fdata = list(zip(new_array_for_allpoints, labels))

    for i in range(len(fdata)):
        for label in labels:
            if (label == fdata[i][1]):
                mid_point_for_lane[label] = fdata[i][0]

    clf = LocalOutlierFactor().fit_predict(
        new_array_for_allpoints)
    outlier_data = list(zip(new_array_for_allpoints, clf))

    for i in outlier_data:
        for j in center_points_cur_frame:
            if i[1] == -1 and i[0] == j:
                cv2.circle(frame, i[0], 50, (0, 0, 0), 3)

    # print(mid_point_for_lane)
    for i in mid_point_for_lane:
        if i > 0:
            cv2.putText(frame, str(
                i), mid_point_for_lane[i], cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # for (centers, label) in zip(new_array_for_allpoints, labels):

    #     if int(label) > 0:
    #         cv2.circle(frame, centers, 7, (int(label)*10,
    #                    int(label)*10, int(label)*10), 3)

    # ms = MeanShift()
    # ms.fit(new_array_for_allpoints)

    b_default = DBSCAN(eps=2, min_samples=2).fit(new_array_for_allpoints)
    # print(labels)

    # for (centers, label) in zip(b_default.components_, b_default.labels_):
    #     please = (int(centers[0]), int(centers[1]))
    #     # print(please)
    #     if int(label) > 0:
    #         cv2.circle(frame, please, 7, (0,
    #                    0, int(label)*10), 3)

    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()
    center_points_tempprev_frame = center_points_temp_frame.copy()

    # prev_first_point = first_og_points.copy()
    # prev_first_point = fp.copy()
    # print(new_array_for_allpoints)

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


# to_store_values = pd.DataFrame(new_array_for_allpoints)
# to_store_values.to_csv('allpoints.csv')
# video.release()
cap.release()
cv2.destroyAllWindows()
