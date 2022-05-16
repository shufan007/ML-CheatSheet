
Linux

修改用户组和用户名称

```
sudo chown -R tdops:users shuangxi.fan
```
查看硬件配置
```
cat /proc/meminfo | grep Mem | awk '{print $1 $2/1024/1024 " GB"}'
```

查看磁盘
```
df -h

sudo du -hd 1
```

查看目录空间占用磁盘大小
```commandline
sudo du --max-depth=1 -h

# 查看目录大小：
du -h --max-depth=1 /nfs/volume-807-1/fanshuangxi/
```

显示隐藏文件：
shift+cmmand+.

环境变量：
```commandline
more /etc/profile
cat ~/.bashrc
```

设置免密登录：
/home/skyler 目录的属主和权限配置不当。然后上网查找资料得知：SSH不希望home目录和~/.ssh目录对组有写权限，于是执行以下命令进行更改：
```commandline
chmod g-w /home/fsx
```
SSH进行认证的过程中除了对用户目录有权限要求外，对.ssh文件夹和authorized_keys文件同样也要限制，如果日志中提示这两个的问题，可以通过如下方式进行修改：
```commandline
chmod 700 /home/fsx/.ssh
chmod 600 /home/fsx/.ssh/authorized_keys
```

### 文件打包，解包
tar
```
解包：tar xvf FileName.tar

打包：tar cvf FileName.tar DirName

压缩文件：

tar -czf file.tar.gz file

解压：

tar -xzvf file.tar.gz
```

### 压缩
```commandline
tar -cvf jpg.tar *.jpg //将目录里所有jpg文件打包成tar.jpg
tar -czf jpg.tar.gz *.jpg //将目录里所有jpg文件打包成jpg.tar后，并且将其用gzip压缩，生成一个gzip压缩过的包，命名为jpg.tar.gz
tar -cjf jpg.tar.bz2 *.jpg //将目录里所有jpg文件打包成jpg.tar后，并且将其用bzip2压缩，生成一个bzip2压缩过的包，命名为jpg.tar.bz2
tar -cZf jpg.tar.Z *.jpg //将目录里所有jpg文件打包成jpg.tar后，并且将其用compress压缩，生成一个umcompress压缩过的包，命名为jpg.tar.Z
rar a jpg.rar *.jpg //rar格式的压缩，需要先下载rar for linux
zip jpg.zip *.jpg //zip格式的压缩，需要先下载zip for linux
```

### 解压
```commandline
tar -xvf file.tar //解压 tar包
tar -xzvf file.tar.gz //解压tar.gz
tar -xjvf file.tar.bz2 //解压 tar.bz2
tar -xZvf file.tar.Z //解压tar.Z
unrar e file.rar //解压rar
unzip file.zip //解压zip
```

### 远程拷贝文件：

本地操作：例如：
```commandline
scp -r admin@10.58.12.8:/home/admin/algo_so_build/turing-algo/src/algo_scorecard/py2so/release release

scp xxx  tdops@10.58.12.147:/data01/shuangxi.fan/
```

二、将远程服务器上的文件复制到本机
```commandline
# scp remote@www.abc.com:/usr/local/sin.sh /home/administrator

sudo scp fanshuangxi@10.57.240.167:~/Downloads/koalas-1.3.0-py3-none-any.whl ./
```

查找文件

```commandline
# 通过find去查找某个文件 

find / -name mysql

find . -name 'resnet18*'
```
which查找命令：通过path环境变量查找执行的文件路径


### Vim 
显示行号

:set number 或者 :set nu然后回车


## git

```commandline
git stash 暂存

git stash pop  恢复

git submodule 更新
git submodule update --init --recursive

撤销commit 或 merge
git revert 【方法二撤销merge时提交的commit的版本号，这里是88edd6d】

恢复某个版本
git reset --hard e273a1a412206da4629a3689aba0e26acad5730a

追加提交
git commit --amend

// 强制推送 (本地分支:远程分支)
git push -f origin master:master

git add 多个文件，但想忽略某一个文件

git add --all -- ':!d.txt'

撤销所有的已经add的文件
git reset HEAD .

git reset --mixed这样文件退出暂存区，但是修改保留
```

git命令	
```commandline
$ git reset --hard HEAD^         回退到上个版本	
$ git reset --hard HEAD~3        回退到前3次提交之前，以此类推，回退到n次提交之前
$ git reset --hard commit_id     退到/进到 指定commit的sha码

```
git如何恢复本地删除的文件夹
```commandline
git reset HEAD  [被删文件夹名称]

git checkout  [ 被删除的文件或文件夹 ]
```

Git pull的强制覆盖
```commandline
$ git fetch --all
$ git reset --hard origin/master 
$ git pull
```

git 查看远程仓库地址
```commandline
git remote -v
```

git 添加下载权限
User Settings 添加Access Tokens
```commandline
# user  automl:xxx
git clone xxx
```

只对github进行代理，对国内的仓库不影响
```commandline
git config --global http.https://github.com.proxy https://127.0.0.1:1080 
git config --global https.https://github.com.proxy https://127.0.0.1:1080

# 同时，如果在输入这条命令之前，已经输入全局代理的话，可以输入进行取消
git config --global --unset http.proxy
git config --global --unset https.proxy

Mac 查看url ip
nslookup https://github.com 
```

解决GitHub下载慢的有效方法:

https://juejin.cn/post/6850037279587565582 

刷新 DNS 缓存
mac执行：sudo dscacheutil –flushcache

查看brew源
```commandline
# brew.git镜像源
git -C "$(brew --repo)" remote -v

# homebrew-core.git镜像源
git -C "$(brew --repo homebrew/core)" remote -v

# homebrew-cask.git镜像源
git -C "$(brew --repo homebrew/cask)" remote -v 
```

### brew 换源

```commandline
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles"
```

## python

查看python安装路径：

方法1：whereis python

查看所有python的路径，不止一个

方法2：which python
查看当前使用的python路径

!python -m torch.utils.collect_env

Pip 安装 指定源

 -i https://pypi.tuna.tsinghua.edu.cn/simple

阿里云 http://mirrors.aliyun.com/pypi/simple/ 
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/ 
豆瓣(douban) http://pypi.douban.com/simple/ 
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/ 
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

查看pip安装列表

pip freeze

sudo vi bin/pip

Pip list  

或 Python pip list

对安装包的更新做限制 Pillow==5.0.0 matplotlib==2.1.2 numpy==1.19.1

./anaconda3-spark/bin/python ./anaconda3-spark/bin/pip install koalas-1.3.0-py3-none-any.whl Pillow==5.0.0 matplotlib==2.1.2 numpy==1.19.1

安装anaconda3 配置环境变量

wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh

anaconda3 自动安装及初始化方法：

sh ./Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda3 \

&& /opt/anaconda3/bin/conda init

source ~/.bashrc

### 安装Anaconda3 示例
```commandline
FROM 10.58.10.201:55000/spark-py:v2.4.7.2

#COPY  .  /home

ENV ANACONDA_SH Anaconda3-2020.02-Linux-x86_64.sh

RUN  wget https://mirrors.bfsu.edu.cn/anaconda/archive/$ANACONDA_SH -P /home/ \

	&& sh /home/$ANACONDA_SH -b -p /opt/anaconda3 \

	&& /opt/anaconda3/bin/conda init \

	&& source ~/.bashrc \

	&& rm /home/$ANACONDA_SH \

	&& pip install hdfs sklearn2pmml lightgbm xgboost
```

```commandline
 docker run -it -p 8888:8888 anaconda3-jupyter /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"

 docker run -it -p 8888:8888 -v D:/work:/opt/notebooks anaconda3-jupyter /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"

```

镜像：fanshuangxi/automl:flaml-env-0.1.1    
测试脚本： ${HOME} /automl_flaml_interface /automl_main.py  
命令示例：
```commandline
python automl_main.py --task_id task_1235 --train_data /home/jovyan/criteo_data_dev/criteo_train --test_data /home/jovyan/criteo_data_dev/criteo_test --label_col label --time_budget 60 --task classification --metric log_loss --estimator_list "['lgbm']" --seed 1

python automl_main.py --task_id task_1235 --train_data /home/jovyan/test_data/criteo_train --test_data /home/jovyan/test_data/criteo_test --label_col label --time_budget 60 --task classification --metric log_loss --estimator_list "['lgbm']" --seed 1

```

### 环境变量
```commandline
export HADOOP_USER_NAME=prod_alita
export HADOOP_USER_PASSWORD=xxx

export JAVA_HOME=/home/jovyan/workspace/automl_test/bigdata_tools/jdk1.8.0_77

export HADOOP_HOME=/home/jovyan/workspace/automl_test/bigdata_tools/didi-hadoop-nmg-2.7.2

os.environ['JAVA_HOME'] = '/home/jovyan/workspace/automl_test/bigdata_tools/jdk1.8.0_77'

os.environ['HADOOP_HOME'] = '/home/jovyan/workspace/automl_test/bigdata_tools/didi-hadoop-nmg-2.7.2'


HOME="/home/submarine"

echo "export JAVA_HOME=$HOME/hadoop_tools/jdk1.8.0_77" >> ~/.bashrc
echo "export HADOOP_HOME=$HOME/hadoop_tools/didi-hadoop-nmg-2.7.2" >> ~/.bashrc
echo "export PATH=$HOME/hadoop_tools/didi-hadoop-nmg-2.7.2/bin:$PATH" >> ~/.bashrc

echo "export CLASSPATH=`/home/submarine/hadoop_tools/didi-hadoop-nmg-2.7.2/bin/hdfs classpath --glob`" >> ~/.bashrc
# echo "export LD_LIBRARY_PATH=$HOME/hadoop_tools/didi-hadoop-nmg-2.7.2/lib/native" >> ~/.bashrc
echo "export ARROW_LIBHDFS_DIR=$HOME/hadoop_tools/didi-hadoop-nmg-2.7.2/lib/native" >> ~/.bashrc

echo "setup success! please run 'source ~/.bashrc' before use"
```

Criteo data 

train_data = "hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet"

test_data = "hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet"

### k8s 提交job
```commandline
kubectl run automl --image=fanshuangxi/automl:latest --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- python /home/submarine/automl_flaml_interface/automl_main.py --task_id test --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 240 --task classification --metric roc_auc --estimator_list "['lgbm','xgboost']"  --seed 1
```

--task_id test --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 240 --task classification --metric roc_auc --estimator_list "['lgbm','xgboost']"  --seed 1


## java

编译java工程
cd interface/java-sdk

mvn package

mvn install

mvn deploy 

执行测试示例：
```commandline
java -cp dml-inference-sdk-example-0.1.1-SNAPSHOT-jar-with-dependencies.jar com.didichuxing.dml.inference.DMLClientApp
```


## hdfs

hdfs查看：
```commandline
hadoop fs -ls /user/turing/algorithm/prod/algoso/
```
上传文件：
```commandline
hadoop fs -put algoso/algo_dc_ml_2c_mlp_so.cpython-36m-x86_64-linux-gnu.so /user/turing/algorithm/prod/algoso/
```
下载文件到本地：
```commandline
hdfs dfs -get /hdfs路径 /本地路径
```

从HDFS中删除文件夹：
```commandline
hadoop fs -rm -r -skipTrash /folder_name
```
