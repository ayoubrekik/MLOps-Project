# Use an official Python image as the base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask and MLflow ports
EXPOSE 5000 5001

# Command to run the Flask app
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5001 & python app.py"]



