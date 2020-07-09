FROM python:3.7.0

RUN pip install --no-cache fire black implicit==0.4.0 scikit-learn scipy tensorflow
RUN pip install --no-cache pytest pandas

COPY . /src
WORKDIR src

RUN useradd xanthus
USER xanthus

CMD ["pytest", "tests"]