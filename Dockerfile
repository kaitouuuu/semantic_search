# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a mount point for the volume
VOLUME /app/local_data

# Make ports 7860 and 7861 available to the world outside this container
EXPOSE 7860 7861

# Run both app.py and data.py when the container launches
CMD ["sh", "-c", "python data.py & python app.py"]
