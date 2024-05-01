# Use the official Python image as the base image
FROM --platform=linux/amd64 python:3.9

# Set the working directory
WORKDIR /app/src

# Install any required packages
RUN apt-get update && apt-get upgrade -y

# Install poetry
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files
COPY app/src/pyproject.toml app/src/poetry.lock /app/src/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without dev

# Copy the source code into the container
COPY app/src /app/src

# Set the default command to run the script
CMD ["echo", "Error: no run specification provided"]
