FROM --platform=linux/amd64 jupyter/scipy-notebook:lab-3.5.3

COPY torch_flwr-requirements.txt ./
RUN pip install --no-cache-dir -r torch_flwr-requirements.txt