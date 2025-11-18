Got you — here is the **actual file**, all in one block so you can copy it straight into `Use_of_AI.md` (or `Use_of_AI.txt`).

---

## Use of AI

[1]. **Tool:** ChatGPT
**Prompt:**
Give me starter Colab code to clone the `coe379l-fa25` GitHub repo, point to the Hurricane Harvey dataset in `datasets/unit03/Project2`, and load the images into a TensorFlow `image_dataset_from_directory` pipeline.
**Output (used):**
- Shell and Python code to:
- clone the repo with `!git clone https://github.com/joestubbs/coe379l-fa25.git`
- set `DATA_ROOT` to `/content/coe379l-fa25/datasets/unit03/Project2`
- create a dataset using `tf.keras.utils.image_dataset_from_directory(...)` with `image_size=(150, 150)` and `batch_size=32`.
- I kept this structure and adapted paths and print statements.

[2]. **Tool:** ChatGPT
**Prompt:**
Show me code in TensorFlow/Keras to preprocess the Hurricane dataset, including normalization, caching, shuffling, prefetching, and a quick visualization of one batch.
**Output (used):**
- Definition of a `Rescaling(1./255)` normalization layer.
- A helper function that maps the normalization layer over each dataset and applies `.cache().shuffle(...).prefetch(tf.data.AUTOTUNE)` to `train_ds`, `val_ds`, and `test_ds`.
- Example loop to take one batch from the dataset and print image and label shapes.
- I used the pattern but adjusted the shuffle buffer and some comments.

[3]. **Tool:** ChatGPT
**Prompt:**
Write a Keras function that builds a simple dense (fully connected) neural network for binary image classification on 150x150x3 inputs, and a helper function to compile and train it.
**Output (used):**
- A `build_dense_ann(input_shape)` function with layers:
- `Flatten()`
- dense layers with ReLU activation
- a final `Dense(1, activation="sigmoid")` output.
- A `compile_and_train(model, train_ds, val_ds, name, epochs=...)` helper that calls `model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])` and trains with `model.fit(...)`, then evaluates on `val_ds`.
- I changed the number of units and epochs, but the overall structure came from this output.

[4]. **Tool:** ChatGPT
**Prompt:**
Provide a Keras implementation of the classic LeNet‑5 CNN adapted to 150x150 RGB images for binary classification (single sigmoid output).
**Output (used):**
- A `build_lenet5_model(input_shape)` function using:
- `Conv2D(6, (5,5), activation="relu")` followed by `AveragePooling2D(pool_size=(2,2))`
- `Conv2D(16, (5,5), activation="relu")` followed by `AveragePooling2D(pool_size=(2,2))`
- `Flatten()`, `Dense(120, activation="relu")`, `Dense(84, activation="relu")`, `Dense(1, activation="sigmoid")`.
- I fixed a small error about `AveragePooling2D` needing `pool_size` and tweaked some details, but the base architecture came from this output.

[5]. **Tool:** ChatGPT
**Prompt:**
Implement an “alternate LeNet‑5” style CNN in Keras with a bit more capacity (more filters and dropout) for the Hurricane damage dataset, and show how to train it.
**Output (used):**
- A `build_alt_lenet5_model(input_shape)` function including:
- `Conv2D(32, (5,5), activation="relu")` → `MaxPooling2D`
- `Conv2D(64, (5,5), activation="relu")` → `MaxPooling2D`
- `Conv2D(128, (3,3), activation="relu")` → `MaxPooling2D`
- `Flatten()`, `Dense(256, activation="relu")`, `Dropout(0.5)`, `Dense(1, activation="sigmoid")`.
- I used this as the starting point and adjusted filter sizes and epoch counts.

[6]. **Tool:** ChatGPT
**Prompt:**
Show me how to train all three models (dense ANN, LeNet‑5, alt‑LeNet‑5), record their validation metrics, select the best model by highest validation accuracy, evaluate it on the test set, and save `models/best_model.keras` and `models/metadata.json`.
**Output (used):**
- Code that:
- Stores results as a list of tuples `(name, model, val_loss, val_acc)`.
- Selects the best model with `max(results, key=lambda r: r[3])`.
- Evaluates the best model on `test_ds`.
- Creates a `models` directory, calls `best_model.save("models/best_model.keras")`, and writes a `metadata.json` file with fields such as `best_model_name`, `img_height`, `img_width`, `val_accuracy`, and `test_accuracy`.
- I reused and modified this logic to fit my variable names.

[7]. **Tool:** ChatGPT
**Prompt:**
Write a Flask-based inference server that loads my saved model from `models/best_model.keras` and metadata from `models/metadata.json`, and exposes two endpoints: GET `/summary` (JSON metadata) and POST `/inference` (binary image → JSON prediction with "damage" or "no_damage").
**Output (used):**
- A `server.py` skeleton that:
- Imports `Flask`, `jsonify`, `request`, TensorFlow, NumPy, `PIL.Image`, and `io`.
- Loads the model using `tf.keras.models.load_model("models/best_model.keras")`.
- Loads metadata using `json.load(open("models/metadata.json"))`.
- Implements `@app.route("/summary", methods=["GET"])` returning model and dataset info as JSON.
- Implements `@app.route("/inference", methods=["POST"])` that reads raw bytes from `request.get_data()`, decodes and resizes the image, normalizes pixel values, runs `MODEL.predict(...)`, and returns `{ "prediction": "damage" }` or `{ "prediction": "no_damage" }`.
- I kept this design and made small adjustments to error handling and print statements.

[8]. **Tool:** ChatGPT
**Prompt:**
Provide a Dockerfile for this Flask/TensorFlow inference server that uses `python:3.11-slim`, installs TensorFlow 2.19, Pillow, NumPy, and Flask, copies `server.py` and the `models` directory into `/app`, exposes port 8000, and runs `python server.py`. I will build the image for `linux/amd64`.
**Output (used):**
- A Dockerfile that:
- Starts from `FROM python:3.11-slim`.
- Sets `WORKDIR /app`.
- Installs OS packages needed by TensorFlow/Pillow (`libglib2.0-0`, `libgl1`, etc.).
- Copies `server.py` and `models/` into the container.
- Installs Python packages with `pip install --no-cache-dir tensorflow==2.19.0 pillow numpy flask`.
- Exposes port `8000`.
- Uses `CMD ["python", "server.py"]`.
- I used this Dockerfile as the basis for my final container.

[9]. **Tool:** ChatGPT
**Prompt:**
Write a `docker-compose.yml` and README instructions that start my Docker Hub image `abhireddy23/hurricane-damage-api:v4` on port 5000 (host) → 8000 (container), and give example `curl` commands for the `/summary` and `/inference` endpoints.
**Output (used):**
- A `docker-compose.yml` file with:
- A service called `hurricane-damage-api`
- `image: abhireddy23/hurricane-damage-api:v4`
- `platform: linux/amd64`
- `ports: ["5000:8000"]`
- README text explaining how to:
- Run `docker compose up -d` to start and `docker compose down` to stop.
- Call `GET /summary` via `curl http://localhost:5000/summary`.
- Call `POST /inference` via
`curl -X POST http://localhost:5000/inference --data-binary "@path/to/test_image.jpg"`.
- Interpret the JSON responses `{ "prediction": "damage" }` or `{ "prediction": "no_damage" }`.
- I used and lightly edited this text in my final README.

---

You can now upload this as your **Use_of_AI** file to GitHub and reference entries like `[3]`, `[7]`, etc. in comments inside your notebook and code.
