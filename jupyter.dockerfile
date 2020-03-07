FROM ubuntu:latest
WORKDIR /home/rg727/HEC_HMS_LSTM
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential python-dev
RUN apt-get install -y python python-distribute python-pip
RUN pip install pip --upgrade
RUN pip install jupyter
COPY requirements.txt /home/rg727/HEC_HMS_LSTM 
ADD ./ /home/rg727/HEC_HMS_LSTM/
RUN pip install -r /home/rg727/HEC_HMS_LSTM/requirements.txt
CMD [ "python", "./Hourly_LSTM.py" ] 
