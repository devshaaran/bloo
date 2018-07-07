import speech_recognition as sr
import pyttsx3
import cv2
import face_recognition
import random as rand
from chatterbot import ChatBot



chatbot = ChatBot(
    'Shaaran',
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

# Train based on the english corpus
chatbot.train("chatterbot.corpus.english")
audio = sr.Recognizer()

while True:
    with sr.Microphone() as micro:
        try:
            eng = pyttsx3.init()
            eng.say("please speak")
            eng.runAndWait()
            initiator = audio.listen(micro,phrase_time_limit=3)
            start_speech = audio.recognize_google(initiator)
            print(start_speech)
            start_test = start_speech.split()
            print(start_test)
            if 'hey' and 'bro' in start_test:

                starting_greet = ['what a great day','ahhhh','boom']
                eng = pyttsx3.init()
                eng.say(rand.choice(starting_greet))
                eng.runAndWait()

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

                no = 0

                while no < 100:

                    ret, frame = video_capture.read()

                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    rgb_small_frame = small_frame[:, :, ::-1]

                    if process_this_frame:

                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

                        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

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

                    #cv2.imshow('Video', frame)

                    no = no + 1

                    if len(face_names) > 0 and face_names[0] in known_face_names:
                        break
                    else:
                        pass

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

                if len(face_names) > 0:
                    eng = pyttsx3.init()
                    eng.say("hi shaaran , what is up for today")
                    eng.runAndWait()
                    break

                else:
                    eng = pyttsx3.init()
                    eng.say("hello there, how may i help you today")
                    eng.runAndWait()
                    break
            else:
                continue


        except Exception:
            continue

with sr.Microphone() as micro:
    while True:
        try:
            source = audio.listen(micro, phrase_time_limit=5)
            x = audio.recognize_google(source)
            print('YOU: ' + str(x))
            ans = chatbot.get_response(str(x))
            print('BOT: ' + str(ans))
            eng = pyttsx3.init()
            eng.say(ans)
            eng.runAndWait()

        except Exception:
            pass

# TODO recognize voice and manipulate for normal speech
# TODO manipulate for operating appliances by speech
# TODO manipulate for operating appliances by text
# TODO make a interactive screen for the bot to interact by giving inputs through voice and face rec

