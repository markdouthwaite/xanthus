FROM python:3.6.8

RUN apt-get update -y
RUN apt-get install gcc musl-dev

COPY . ./app
WORKDIR app

RUN pip install -r requirements.txt

RUN python setup.py build_ext --inplace
RUN python setup.py install

CMD ["pytest", "tests"]