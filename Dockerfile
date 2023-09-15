FROM public.ecr.aws/lambda/python:3.8

COPY model.py ${LAMBDA_TASK_ROOT}
COPY model.bin ${LAMBDA_TASK_ROOT}

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]