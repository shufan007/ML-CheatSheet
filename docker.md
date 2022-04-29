
## Docker 命令

### 镜像

搜索镜像：
```
docker search keyword
```

下载镜像：
```
docker pull 镜像
```

查看镜像：
```
Docker images
```

保存镜像为文件:
```
docker save -o 要保存的文件名  要保存的镜像
# or
docker save d3949e34634c > /home/centos7.4.tar
```

从文件载入镜像:
```
docker load < 文件名
```

通过Dockerfile打镜像:
```
docker build -t spark-py:v2.4.7.4 .
```

修改tag：
```
docker tag 8e789a3e5f4f fanshuangxi:automl-flaml-jupyter-0.1
```

删除tag:
```
docker rmi -f hub.doge.com/ubuntu:latest
```

删除镜像:
```
docker rmi 容器id
docker rmi -f 容器id
```

ctrl+P+Q可退出docker，重新进入docker使用如下命令

```
sudo docker container exec -it < ppocr /bin/bash >
```

Docker查看磁盘使用
```
docker system df
```
清理磁盘，删除关闭的容器、无用的数据卷和网络，以及dangling镜像(即无tag的镜像)
```
docker system prune
```

### 容器


运行容器：
```
docker run -it 容器名
```

以root进入
```
docker run -u root -it dc0303bd9d78
```

Docker 拷贝文件到容器:
```
docker cp tf_test_model_save.py b31b9c1faed2:/home/
```

进入正在运行中的容器：
```
sudo docker attach 容器ID
```

### 目录挂载：
```
docker run -p 8500:8500 --mount type=bind,source=/tmp/mnist,target=/models/mnist  -e MODEL_NAME=mnist -t tensorflow/serving &

docker run -p 8500:8500 --mount type=bind,source=/data01/shuangxi.fan/docker/mnist/models/mnist_model,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving

docker run -d -p 8500:8500 -v source=/Users/fanshuangxi/docker/mnist/models/mnist_save_model,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving &

# Jupiter:

docker run -p 8888:8888 -v ~/Projects/AutoML:/home/jovyan  -t fanshuangxi/automl:flaml-jupyter-0.1

127.0.0.1:8888
```

### 将容器打包成镜像
```
docker commit -a "shuangxi.fan" -m "tensorflow bert" b31b9c1faed2 tensorflow/tensorflow_nlp:1.0
```

其中，-m 来指定提交的说明信息，跟我们使用的版本控制工具一样；-a 可以指定更新的用户信息；之后是用来创建镜像的容器的 ID；最后指定目标镜像的仓库名和 tag 信息。创建成功后会返回这个镜像的 ID 信息。
```
docker commit -a "shuangxi.fan" -m “automl-flaml” 081ca5359930 fanshuangxi/automl:latest
```

### 删除镜像

docker要删除镜像，先要删除依赖它的容器

1. 删除容器
```
docker ps #查看正在运行的容器

docker ps -a #查看所有容器

docker rm container_id #删除容器
```
2. 删除镜像
```
docker images //查看镜像

docker rmi image_id
```
2.1 删除其他镜像
```
# 删除 null image

sudo docker rmi $(docker images -f "dangling=true" -q) #删除所有镜像

# 删掉容器

docker stop $(docker ps -qa)

docker rm $(docker ps -qa)

# 删除镜像

docker rmi --force image_id

docker rmi --force $(docker images -q)

# 删除名称中包含某个字符串的镜像

# 例如删除包含“some”的镜像

docker rmi --force $(docker images | grep some | awk '{print $3}')
```

删除所有未运行的容器（已经运行的删除不了，未运行的就一起被删除了）
```
sudo docker rm $(sudo docker ps -a -q)

docker ps -a 显示所有的容器，包括未运行的。

docker ps -q :静默模式，只显示容器编号。

# Docker 1.13版本以后，可以使用 docker container prune 命令，删除孤立的容器。

sudo docker container prune
```

### 上传镜像值hub仓库

使用docker hub的用户名，密码登录即可
```
docker login

docker push fanshuangxi/automl:latest
```

如果上传的镜像不是你的dockerhub和镜像名：Docker ID/仓库名，就先用：docker tag 镜像ID Docker ID/仓库名:新的标签名(tag)
```
docker tag 6da93e7f73bc fanshuangxi/automl:jupyter-notebook-0.6.0-0.1

docker push fanshuangxi/automl:jupyter-notebook-0.6.0-0.1
```



