# Use the official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Replace "requirements.txt" with the actual name of your requirements file
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define an environment variable
# You can set the PORT variable as needed for your script
ENV PORT=50051

# Define The GRC Controller to start with server
# It has [MULTI, LINE, NEIGHBORS, POLYNOMIAL, SVM]
ENV GRPCKEY = MULTI

# Run your Python script
CMD [ "python", "main.py" ]