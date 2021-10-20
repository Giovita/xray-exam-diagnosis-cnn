# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* xray/*.py

black:
	@black scripts/* xray/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist *.dist-info *.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GCP AI PLATFORM
# ----------------------------------

##### GCP Commands  - - - - - - - - - - - - - - - - - - - -

# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
# LOCAL_PATH=PATH_TO_FILE_train_1k.csv

# project id
PROJECT_ID=xray-cnn-329114

# bucket name
BUCKET_NAME=images-xray-lewagon

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=data

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=us-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

##### GCP Params  - - - - - - - - - - - - - - - - - - - - -

BUCKET_TRAINING_FOLDER = 'trainings'

##### Machine configuration - - - - - - - - - - - - - - - -

# REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=TensorFlow
RUNTIME_VERSION=2.5

MACHINE_TYPE=n1-standard-16

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=xray
FILENAME=trainer
FILENAME_BINARY = run_binary
FILENAME_MULTILABEL = run_multilabel

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=xray_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
    --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
    --staging-bucket=gs://${BUCKET_NAME} \
		--package-path=${PACKAGE_NAME} \
		--module-name=${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
    --scale-tier=BASIC_GPU \
		--stream-logs

gcp_submit_training_binary:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
    --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
    --staging-bucket=gs://${BUCKET_NAME} \
		--package-path=${PACKAGE_NAME} \
		--module-name=${PACKAGE_NAME}.${FILENAME_BINARY} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
    --scale-tier=BASIC_GPU \
		--stream-logs

gcp_submit_training_multilabel:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
    --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
    --staging-bucket=gs://${BUCKET_NAME} \
		--package-path=${PACKAGE_NAME} \
		--module-name=${PACKAGE_NAME}.${FILENAME_MULTILABEL} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
    --scale-tier=BASIC_GPU \
		--stream-logs


gcp_submit_training_both:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
    --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
    --staging-bucket=gs://${BUCKET_NAME} \
		--package-path=${PACKAGE_NAME} \
		--module-name=${PACKAGE_NAME}.${FILENAME_MULTILABEL} \
		--module-name=${PACKAGE_NAME}.${FILENAME_BINARY} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
    --scale-tier=BASIC_GPU \
		--stream-logs

# --master-machine-type ${MACHINE_TYPE}
# ----------------------------------
#      Prediction API
# ----------------------------------
run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
