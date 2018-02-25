import face_recognition
import cv2
import math



def face_Recognition():
    video_capture = cv2.VideoCapture(0)

    shaaran_image = face_recognition.load_image_file("myphoto1.jpg")
    shaaran_encoding = face_recognition.face_encodings(shaaran_image)[0]

    known_face_encodings = [
        shaaran_encoding
    ]
    known_face_names = [
        "Shaaran"
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:

        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

            print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

            for face_landmarks in face_landmarks_list:

                facial_features = ['top_lip']

                f_feat = []
                num = [0, 6]

                for facial_feature in facial_features:
                    for i in num:
                        print("The {} in this face has the following points: {}".format(facial_feature,
                                                                                        face_landmarks[facial_feature]))
                        (x, y) = face_landmarks[facial_feature][i]
                        f_feat.append((x, y))

                [(l_eyeb_x, l_eyeb_y), (r_eyeb_x, r_eyeb_y)] = f_feat
                d1 = (((l_eyeb_x) - r_eyeb_x) ** 2 + (l_eyeb_y - r_eyeb_y) ** 2)
                dl = math.sqrt(d1)
                print(dl)

            face_names = []
            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, '*' + name, (left + 6, bottom - 6), font, 1.3, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# TODO recognize voice and manipulate for normal speech
# TODO manipulate for operating appliances by speech
# TODO manipulate for operating appliances by text
# TODO make a interactive screen for the bot to interact by giving inputs through voice and face rec

