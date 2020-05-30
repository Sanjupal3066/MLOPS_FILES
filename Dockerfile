FROM centos:latest

RUN yum install sudo -y
RUN sudo yum install wget -y
RUN yum install git -y

RUN yum install python36 -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN yum install -y epel-release
RUN yum groupinstall "development tools" -y
RUN yum install -y python36-devel


RUN pip3 install keras
RUN pip3 install --upgrade tensorflow
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install pillow

WORKDIR /project
