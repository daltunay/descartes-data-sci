# Instructions

1. Clone the repository:

   ```bash
   git clone git@github.com:daltunay/descartes-data-sci.git
   ```

2. Navigate into the cloned repository:

   ```bash
   cd descartes-data-sci
   ```

3. Build the Docker image:

   ```bash
   docker build -t app .
   ```

4. Run the Docker container:

   ```bash
   docker run -p 8501:8501 app
   ```

5. Access the Streamlit app in your web browser:

   [localhost:8501](http://localhost:8501/)
