import cv2
import numpy as np
import face_recognition

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face encoding and name
known_image = face_recognition.load_image_file("isabella.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_name = "Isabella"

# Debug print statements
print(f"Known face name: {known_face_name}")
print(f"Number of face encodings: {len(known_face_encoding)}")

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Debug print statement
    #print(f"Found {len(face_locations)} face(s)")

    # Loop through each face found in the frame
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Compare face with known face
        match = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.35) #0.55
        
        # Get name of best match
        if match[0]:
            name = known_face_name
        else:
            name = "Unknown"
        
        # Draw rectangle around face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw label with name below rectangle
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
