import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

claret_image = face_recognition.load_image_file("claret.jpg")
claret_face_encoding = face_recognition.face_encodings(claret_image)[0]

hari_image = face_recognition.load_image_file("hari.jpg")
hari_face_encoding = face_recognition.face_encodings(hari_image)[0]

aravinth_image = face_recognition.load_image_file("aravinth.jpg")
aravinth_face_encoding = face_recognition.face_encodings(aravinth_image)[0]


viancy_image = face_recognition.load_image_file("viancy.jpg")
viancy_face_encoding = face_recognition.face_encodings(viancy_image)[0]


janani_image = face_recognition.load_image_file("janani.jpg")
janani_face_encoding = face_recognition.face_encodings(janani_image)[0]

ramya_image = face_recognition.load_image_file("ramya.jpg")
ramya_face_encoding = face_recognition.face_encodings(ramya_image)[0]



prabha_image = face_recognition.load_image_file("prabha.jpg")
prabha_face_encoding = face_recognition.face_encodings(prabha_image)[0]

deepa_image = face_recognition.load_image_file("deepa.jpg")
deepa_face_encoding = face_recognition.face_encodings(deepa_image)[0]

shobana_image = face_recognition.load_image_file("shobana.jpg")
shobana_face_encoding = face_recognition.face_encodings(shobana_image)[0]


bemesha_image = face_recognition.load_image_file("bemesha.jpg")
bemesha_face_encoding = face_recognition.face_encodings(bemesha_image)[0]


sherril_image = face_recognition.load_image_file("sherril.jpg")
sherril_face_encoding = face_recognition.face_encodings(sherril_image)[0]

kavitha_image = face_recognition.load_image_file("kavitha.jpg")
kavitha_face_encoding = face_recognition.face_encodings(kavitha_image)[0]


anitha_image = face_recognition.load_image_file("anitha.jpg")
anitha_face_encoding = face_recognition.face_encodings(anitha_image)[0]


juliana_image = face_recognition.load_image_file("juliana.jpg")
juliana_face_encoding = face_recognition.face_encodings(juliana_image)[0]

venkatesh_image = face_recognition.load_image_file("venkatesh.jpg")
venkatesh_face_encoding = face_recognition.face_encodings(venkatesh_image)[0]

kumaran_image = face_recognition.load_image_file("kumaran.jpg")
kumaran_face_encoding = face_recognition.face_encodings(kumaran_image)[0]

marshal_image = face_recognition.load_image_file("marshal.jpg")
marshal_face_encoding = face_recognition.face_encodings(marshal_image)[0]

principal_image = face_recognition.load_image_file("principal.jpg")
principal_face_encoding = face_recognition.face_encodings(principal_image)[0]

rexy_image = face_recognition.load_image_file("rexy.jpg")
rexy_face_encoding = face_recognition.face_encodings(rexy_image)[0]

justin_image = face_recognition.load_image_file("justin.jpg")
justin_face_encoding = face_recognition.face_encodings(justin_image)[0]

director_image = face_recognition.load_image_file("director.jpg")
director_face_encoding = face_recognition.face_encodings(director_image)[0]

pd_sir_image = face_recognition.load_image_file("pd_sir.jpg")
pd_sir_face_encoding = face_recognition.face_encodings(pd_sir_image)[0]

merlin_image = face_recognition.load_image_file("merlin.jpg")
merlin_face_encoding = face_recognition.face_encodings(merlin_image)[0]

kavya_CNSI_image = face_recognition.load_image_file("Kavya_CNSI.jpg")
kavya_CNSI_face_encoding = face_recognition.face_encodings(kavya_CNSI_image)[0]

balaji_image = face_recognition.load_image_file("balaji_cnsi.jpg")
balaji_face_encoding = face_recognition.face_encodings(balaji_image)[0]

mathi_image = face_recognition.load_image_file("mathi1.jpg")
mathi_face_encoding = face_recognition.face_encodings(mathi_image)[0]








# Create arrays of known face encodings and their names
known_face_encodings = [
    claret_face_encoding,
    hari_face_encoding,
    aravinth_face_encoding,
    viancy_face_encoding,
    janani_face_encoding,
    ramya_face_encoding,
    prabha_face_encoding,
    deepa_face_encoding,
    shobana_face_encoding,
    bemesha_face_encoding,
    sherril_face_encoding,
    kavitha_face_encoding,
    anitha_face_encoding,
    juliana_face_encoding,
    venkatesh_face_encoding,
    kumaran_face_encoding,
    marshal_face_encoding,
    principal_face_encoding,
    rexy_face_encoding,
    justin_face_encoding,
    director_face_encoding,
    pd_sir_face_encoding,
    merlin_face_encoding,
    kavya_CNSI_face_encoding,
    balaji_face_encoding,
    mathi_face_encoding
    
 
]
known_face_names = [
    "Claret Ivin",
    "Hariharan",
    "Aravinth",
    "Ms Viancy",
    "Dr Janani",
    "Ms Ramya",
    "Ms Prabha",
    "Dr Deepa",
    "Ms Shobana",
    "Ms Bemesha",
    "Ms Sherril",
    "Ms Kavitha",
    "Ms Anitha",
    "Dr Juliana",
    "Mr Venkatesh",
    "Mr Kumaran",
    "Mr Marshal",
    "Principal",
    "Dr Inba Rexy",
    "Rev. Fr. Justine",
    "Director",
    "Dr Benkins",
    "Dr Sengole Merlin",
    "MS Kavya Krishnan",
    "Mr Balaji",
    "Anbu Mathi"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
