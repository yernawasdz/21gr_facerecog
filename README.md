# Face Recognition API – Hackathon Project (Team 21gr)

A full-stack facial recognition system designed to store, recognize, and retrieve faces from uploaded images. Built during a hackathon and awarded 3rd place.  
This project demonstrates practical use of computer vision, face matching, and secure API-based communication.

---

## Features

- Upload one or multiple images and extract individual faces  
- Recognize uploaded faces against an existing database  
- Retrieve specific face images by ID  
- Authentication via OAuth2 (JWT tokens)  
- Frontend with login and API interaction  
- Video walkthrough available in Russian

---

## Demo Video

**Watch on YouTube:** [https://youtu.be/kXas9sNMTeo](https://youtu.be/kXas9sNMTeo)

---

## How to Run

Make sure you have Python and `uvicorn` installed, then run:

```bash
uvicorn main:app --reload
```
The app will be available at: http://127.0.0.1:8000

Authentication Required
All API endpoints require login.

Demo Login Credentials
These credentials are already registered in the system:
```
Username: user@example.com  
Password: $2b$12$examplehashedpassword

```

## Frontend Overview
- The frontend is implemented using HTML and JavaScript. It supports:
- User login and token retrieval
- Uploading images and saving faces
- Recognizing uploaded faces
- Fetching images by ID
- All API calls include the access token in the Authorization header.
- Accuracy and Testing
- 5 base face images stored. 64 test images used for recognition

Recognition accuracy: 92.12% (59 out of 64 images matched correctly)

## Tech Stack
- Backend: FastAPI with face_recognition
- Frontend: HTML and Vanilla JavaScript
- Authentication: OAuth2 with JWT
- Deployment: Uvicorn
- Storage: Face embeddings stored on disk or in database

