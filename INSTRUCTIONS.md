# Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/daltunay/descartes-data-sci.git
   ```

2. Navigate into the cloned repository:

   ```bash
   cd descartes-data-sci
   ```

In order to run the app, you have two possibilities: either with via Docker or poetry.

## Docker

3. Build the Docker image:

   ```bash
   docker build -t app .
   ```

4. Run the Docker container:

   ```bash
   docker run -p 8501:8501 app
   ```

## Poetry

Optional: install `poetry` via `pip install poetry`

3. Install dependencies:

   ```bash
   poetry install --no-root
   ```

4. Run the app:

   ```bash
   poetry run streamlit run app.py
   ```

---

5. Access the Streamlit app in your web browser:

   [localhost:8501](http://localhost:8501/)
