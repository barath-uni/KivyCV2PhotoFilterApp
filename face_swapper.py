from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import pytesseract
from kivymd.uix.label import MDLabel
import dlib
import numpy as np


class MainApp(MDApp):
    indexes_triangles = []

    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        self.label = MDLabel()
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_facade = cv2.CascadeClassifier("haarcascade_smile.xml")
        self.save_img_button = MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.save_img_button.bind(on_press=self.take_picture)
        layout.add_widget(self.save_img_button)
        self.parse_image()
        self.capture = cv2.VideoCapture(2)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def parse_image(self):
        self.img = cv2.imread("rajini.jpg")
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(self.img_gray)

        # Face 1
        faces = self.detector(self.img_gray)
        for face in faces:
            landmarks = self.predictor(self.img_gray, face)
            self.landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                self.landmarks_points.append((x,y))
            points = np.array(self.landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(self.mask, convexhull, 255)

            # delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(self.landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, np.int32)

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = self.extract_index_nparray(index_pt1)

                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = self.extract_index_nparray(index_pt2)

                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = self.extract_index_nparray(index_pt3)

                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    self.indexes_triangles.append(triangle)
        cv2.imshow("rajini_mask", self.mask)


    def extract_index_nparray(self, nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    def load_video(self, *args):
        global convexhull2
        ret, img2 = self.capture.read()
        # Frame initialize
        self.image_frame = img2
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)

        # Face 2
        faces2 = self.detector(img2_gray)
        if faces2:
            for face in faces2:
                landmarks = self.predictor(img2_gray, face)
                landmarks_points2 = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points2.append((x, y))
                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)


            # Triangulation of both faces
            for triangle_index in self.indexes_triangles:
                # Triangulation of the first face
                tr1_pt1 = self.landmarks_points[triangle_index[0]]
                tr1_pt2 = self.landmarks_points[triangle_index[1]]
                tr1_pt3 = self.landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = self.img[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)

                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                # Triangulation of second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

            # Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(img2_gray)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)

            img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, img2_new_face)

            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

            seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
            buffer = cv2.flip(seamlessclone, 0).tostring()
            texture = Texture.create(size=(seamlessclone.shape[1], seamlessclone.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def take_picture(self, *args):
        image_name = "picture_at_2_02.png"
        img = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        text_data = pytesseract.image_to_string(img, lang='eng', config="--oem 3 --psm 6")
        print(text_data)
        self.label.text = text_data
        cv2.imshow("cv2 final image", img)
        cv2.imwrite(image_name, self.image_frame)


if __name__ == '__main__':
    MainApp().run()
