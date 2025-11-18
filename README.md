# Hurricane Damage Inference Server (Part 3)

This directory contains the files for Part 3 of the project: a Dockerized inference server for classifying satellite images of buildings as either **damage** or **no_damage** after Hurricane Harvey.

The server exposes two HTTP endpoints:

* `GET /summary` – returns JSON metadata about the model.
* `POST /inference` – accepts raw image bytes and returns JSON with a top-level `prediction` field whose value is either `"damage"` or `"no_damage"`.

These endpoints are designed to match the project specification and work with the provided grader scripts (`grader.py` and `start_grader.sh`).

---

## Docker Hub Image

* Image name: `abhireddy23/hurricane-damage-api:v4`
* Architecture: `linux/amd64` (x86)

Pull command:

docker pull abhireddy23/hurricane-damage-api:v4

---

## Files in this directory

* `server.py`
  HTTP server that:

  * Loads `models/best_model.keras` and `models/metadata.json`.
  * Implements:

    * `GET /summary`
    * `POST /inference`

* `Dockerfile`
  Builds the Docker image by:

  * Starting from `python:3.11-slim`.
  * Installing TensorFlow 2.19, Pillow, and NumPy.
  * Copying `server.py` and the `models/` directory.
  * Exposing port `8000`.
  * Running `python server.py`.

* `docker-compose.yml`
  Starts the inference server container using the prebuilt image:

  services:
  hurricane-damage-api:
  image: abhireddy23/hurricane-damage-api:v4
  platform: linux/amd64
  ports:
  - "5000:8000"

  This maps host port 5000 → container port 8000.

* `models/`

  * `best_model.keras` – trained best model from Part 2.
  * `metadata.json` – metadata such as image height/width used for preprocessing.

---

## Running the inference server with docker-compose

From this directory (the one containing `docker-compose.yml`), run:

docker compose up -d

This will:

* Pull `abhireddy23/hurricane-damage-api:v4` if it is not already present.
* Start the container in detached mode.
* Expose the HTTP API on:

[http://localhost:5000](http://localhost:5000)

To stop the server:

docker compose down

---

## API Endpoints and Example Requests

### 1. GET /summary

Returns a JSON object with basic information about the model, such as:

* Model name
* Input/output shapes
* Image dimensions
* Any additional metadata fields

Example request:

curl [http://localhost:5000/summary](http://localhost:5000/summary)

---

### 2. POST /inference

Accepts a binary image payload and returns a JSON object with a top-level `prediction` field.

Request body: raw image bytes (e.g., JPEG or PNG).

Response JSON will be one of:

{ "prediction": "damage" }

or

{ "prediction": "no_damage" }

Example request (JPEG image):

curl -X POST [http://localhost:5000/inference](http://localhost:5000/inference) 
--data-binary "@path/to/test_image.jpg"

These exact string values for `prediction` are what the project grader expects.

---

## Using the provided grader

The course repository includes a `grader.py` and `start_grader.sh` script to automatically check that the inference server conforms to the specification.

1. Make sure the inference server is running on port 5000:

   cd /path/to/this/directory
   docker compose up -d

2. In the directory containing `start_grader.sh` and `grader.py`, run:

   chmod +x start_grader.sh
   ./start_grader.sh

The grader will:

* Send requests to `GET /summary` and `POST /inference`.
* Verify that the endpoints exist and return correctly formatted JSON.
* Check that the `prediction` field is either `"damage"` or `"no_damage"`.

---

## Direct docker run (alternative to docker-compose)

Instead of `docker compose`, the server can also be started directly with `docker`:

docker run -d 
--platform linux/amd64 
-p 5000:8000 
--name hurricane-damage-api 
abhireddy23/hurricane-damage-api:v4

To stop and remove the container:

docker stop hurricane-damage-api
docker rm hurricane-damage-api


