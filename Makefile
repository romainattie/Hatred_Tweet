# project id
PROJECT_ID=tweet-detection
DOCKER_IMAGE_NAME=tweet_image

# bucket name
BUCKET_NAME=wagon-data-779-attie
REGION=europe-west1


docker_build:
	docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run_locally:
	docker run -e PORT=8000 -p 8080:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}
# -e = création de variable d'environnement qui vaut 8000

docker_push_gcr:
	docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Hatred_Tweet/*.py

black:
	@black scripts/* Hatred_Tweet/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Hatred_Tweet-*.dist-info
	@rm -fr Hatred_Tweet.egg-info

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
