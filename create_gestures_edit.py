import cv2
import numpy as np
import pickle
import os
import sqlite3
import random

image_x, image_y = 50, 50

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    # create the folder and database if not exist
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = """
            CREATE TABLE gesture (
                g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                g_name TEXT NOT NULL
            )
        """
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        conn.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to update the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
            conn.execute(cmd, (g_name, g_id))
        else:
            print("No changes made.")
            conn.close()
            return
    conn.commit()
    conn.close()

def store_images(g_id):
    total_pics = 1200
    hist = get_hand_hist()

    # --- improved camera initialization ---
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: cannot open any camera. Exiting.")
        return
    print("Camera opened successfully. Press 'c' to start/stop capturing, 'q' to quit.")

    x, y, w, h = 300, 100, 300, 300
    save_dir = f"gestures/{g_id}"
    create_folder(save_dir)

    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting loop.")
            break

        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0,1], hist, [0,180,0,256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (5,5), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh,)*3)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]

        # find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if contours:
            # pick the largest valid contour
            valid_contours = [cnt for cnt in contours if cnt is not None and len(cnt) >= 3]
            if valid_contours:
                contour = max(valid_contours, key=lambda cnt: cv2.contourArea(np.array(cnt, dtype=np.int32)))
                # ensure correct type & shape
                contour = np.array(contour, dtype=np.int32)
                if contour.ndim == 2:
                    contour = contour.reshape(-1, 1, 2)

                if cv2.contourArea(contour) > 10000 and frames > 50:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    pic_no += 1

                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    # pad to square
                    if w1 > h1:
                        pad = (w1 - h1) // 2
                        save_img = cv2.copyMakeBorder(save_img, pad, pad, 0, 0,
                                                      cv2.BORDER_CONSTANT, (0,0,0))
                    elif h1 > w1:
                        pad = (h1 - w1) // 2
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, pad, pad,
                                                      cv2.BORDER_CONSTANT, (0,0,0))

                    save_img = cv2.resize(save_img, (image_x, image_y))
                    # random flip augmentation
                    if random.randint(0, 1) == 0:
                        save_img = cv2.flip(save_img, 1)

                    cv2.putText(img, "Capturing...", (30, 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (127,255,255), 2)
                    cv2.imwrite(f"{save_dir}/{pic_no}.jpg", save_img)

        # draw ROI and counter
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, str(pic_no), (30, 400),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127,127,255), 2)

        cv2.imshow("Capturing gesture", img)
        cv2.imshow("thresh", thresh)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0
        elif keypress == ord('q'):
            break

        if flag_start_capturing:
            frames += 1
        if pic_no >= total_pics:
            print("Finished capturing all images.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_create_folder_database()
    g_id = input("Enter gesture no.: ").strip()
    g_name = input("Enter gesture name/text: ").strip()
    store_in_db(g_id, g_name)
    store_images(g_id)
