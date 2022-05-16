## K8s

https://kubernetes.io/zh/docs/home/ 

https://kubernetes.io/zh/docs/tutorials/kubernetes-basics/deploy-app/deploy-interactive/ 

https://kubernetes.io/zh/docs/tutorials/hello-minikube/ 

使用 kubeadm 创建集群
https://kubernetes.io/zh/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/ 

kubectl 备忘单
https://kubernetes.io/zh/docs/reference/kubectl/cheatsheet/ 

摆脱 AI 生产“小作坊”：如何基于 Kubernetes 构建云原生 AI 平台
https://zhuanlan.zhihu.com/p/504047864

~/.kube/config

### k8s集群配置

将k8s集群配置yml文件内容放入 ~/.kube/config

~/.kube/config

brew install kubectl 

kubectl version --client
kubectl cluster-info

kubectl get pods                              # 列出当前命名空间下的全部 Pods

brew install helm

Installation Ray Cluster
```commandline
# Navigate to the directory containing the chart
$ cd ray/deploy/charts

# Install a small Ray cluster with the default configuration
# in a new namespace called "ray". Let's name the Helm release "example-cluster."
$ helm -n ray install automl-cluster --create-namespace ./ray

# You can use helm upgrade to adjust the fields minWorkers and maxWorkers without restarting the Ray cluster.

helm -n ray upgrade automl-cluster ./ray
```
### Cleanup
```commandline
# First, delete the RayCluster custom resource.
$ kubectl -n ray delete raycluster example-cluster
raycluster.cluster.ray.io "example-cluster" deleted

# Delete the Ray release.
$ helm -n ray uninstall example-cluster
release "example-cluster" uninstalled

# Optionally, delete the namespace created for our Ray release.
$ kubectl delete namespace ray
namespace "ray" deleted
```


对于单独部署的 ray-operator 可能是以deployment 部署，删除的时候需要以deployment 删除
```commandline
#查看
kubectl get deployment 

kubectl delete deployments ray-operator

```

### 解决K8S的namespace无法删除问题
```commandline
# 查看命名空间
kubectl get namespaces -A

# 列出你的命名空间下有哪些资源没有删除
kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n <你的命名空间>
# 添加补丁：
kubectl -n <你的命名空间> patch <红框内复制过来> -p '{"metadata":{"finalizers":[]}}' --type='merge'

# 如：
kubectl -n ray-d1 patch raycluster.cluster.ray.io/automl-cluster -p '{"metadata":{"finalizers":[]}}' --type='merge'

kubectl -n ray-d1 patch secret/ray-operator-serviceaccount-token-nnjhf -p '{"metadata":{"finalizers":[]}}' --type='merge'

# 再次查看命名空间

```
#### 删除CRD
```commandline
kubectl get crd |grep ray
kubectl delete crd rayclusters.cluster.ray.io
```
#### 删除 serviceAccount
```commandline
kubectl get serviceAccounts |grep ray
# kubectl get serviceaccounts/ray-operator-serviceaccount -o yaml
kubectl delete serviceaccounts/ray-operator-serviceaccount
```

#### 查看 Deployment
```commandline
kubectl get deployment --all-namespaces

```

#### 查看跑起来的容器有哪些
kubectl get pod --all-namespaces


### Running multiple Ray clusters
install the Operator and two Ray Clusters in three separate Helm releases
#### 注：经实验，ray operator 必须单独安装，并且命名空间只能是默认的default，否则无法成功安装多集群
Install the operator in its own Helm release.

```commandline
# Install the operator in its own Helm release.
$ helm install ray-operator --set operatorOnly=true ./ray

# Install a Ray cluster in a new namespace "ray".
$ helm -n ray install example-cluster --set clusterOnly=true ./ray --create-namespace

# Install a second Ray cluster. Launch the second cluster without any workers.
$ helm -n ray install example-cluster2 \
    --set podTypes.rayWorkerType.minWorkers=0 --set clusterOnly=true ./ray
```

### Running Ray cluster dynamic
```commandline
# 首先安装 operator 并创建命名空间，运行在单独的helm release
helm -n ray install ray-operator --create-namespace --set operatorOnly=true ./ray

# 然后安装集群
helm -n ray install automl-cluster1 --set clusterOnly=true --set podTypes.rayHeadType.memory=16 --set podTypes.rayWorkerType.minWorkers=2 --set podTypes.rayWorkerType.memory=16  --set podTypes.rayWorkerType.CPU=2 ./ray

# 向集群提交job
# 可通过yaml文件提交，或命令行提交
# kubectl create -nray -f automl-ray-d.yaml

kubectl -n ray run automl-cluster1-job1 --image=rayproject/ray:latest --env="RAY_ADDRESS=automl-cluster1-ray-head:8265" -- /bin/bash -c "git clone http://didi-automl:xxx@git.xiaojukeji.com/dml/dml-autotabular && cd dml-autotabular && ray job submit --runtime-env-json='{\"working_dir\":\"./\", \"env_vars\": {\"HADOOP_USER_NAME\": \"prod_alita\", \"HADOOP_USER_PASSWORD\": \"xxx\"}}' -- bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric roc_auc --estimator_list '[\"lgbm\"]' --estimator_kwargs '{\"n_jobs\":4,\"n_concurrent_trials\":1}'  --seed 1"

```


```commandline
kubectl describe pods kubectl -n ray get pods
kubectl describe pods ray-operator-5b985c9d77-wdt6d
```

查看pods
```commandline
kubectl describe pods calico-node-9l2b4 -n kube-system

kubectl describe pods -n ray example-cluster-ray-worker-type-s46vm
```

https://hub.docker.com/r/rayproject/ray
```commandline
kubectl logs  $(kubectl get pod -l cluster.ray.io/component=operator -o custom-columns=:metadata.name) | tail -n 100

```

查看pod
```commandline
kubectl -n ray describe pod example-cluster-ray-head-type-w5pmd
```

查看pod日志
```commandline
kubectl -n ray logs -f automl-test-job3-9flqt

kubectl logs -f  etcd-kmaster1 -n kube-system

kubectl logs --tail 200 -f kube-apiserver -n kube-system  #查看最后200行的日志

kubectl logs -l app=frontend # 返回所有标记为 app=frontend 的 pod 的合并日志。

kubectl logs --since=1h nginx#查看名称为nginx这个pod最近一小时的日志

```

进入容器
```commandline
kubectl -n ray exec -it example-cluster-ray-head-type-w5pmd -c ray-node -- /bin/bash

# 查看Pod里业务容器
kubectl get pods myapp-pod -o jsonpath={.spec.containers[*].name}

# 如何运行一个不会关机的shard
kubectl run -i --tty ray-test01 --image=rayproject/ray:latest --restart=Always
kubectl -n ray-test run -i --tty test-node1 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --restart=Always
kubectl -n ray-test run -i --tty test-node4 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --overrides='{ "spec": { "restartPolicy": "Never", "containers": [ { "name":"ray", "image":"fanshuangxi/automl:ray-flaml-cpu-0.1.3", "imagePullPolicy": "Always", "env": [{"name": "HADOOP_USER_NAME","value": "prod_alita"}, {"name": "HADOOP_USER_PASSWORD","value": "xxx"}], "command": ["/bin/bash", "-c", "--"], "args": [ "" ], "resources": { "requests": { "cpu": 4, "memory": "16Gi" } } } ] } }' -- /bin/bash

# 重新进入Running中的容器
# resume using 'kubectl attach test-node2 -c test-node2 -i -t' command when the pod is running
kubectl -n ray-test attach test-node2 -c test-node2 -i -t


# 如何将外部文件复制进入k8s容器内部
kubectl -n ray-test cp ./example-full.yaml test-node1:/home/ray/

#假如当前pod只有一个容器,运行以下命令即可
kubectl exec -it nginx-56b8c64cb4-t97vb -- /bin/bash

假如一个pod里有多个容器,用--container or -c 参数。例如:假如这里有个Pod名为my-pod,这个Pod有两个容器,分别名为main-app 和 helper-app,下面的命令将打开到main-app的shell的容器里。

```

删除job
```commandline
kubectl get jobs -n ray

kubectl delete job ray-test-job1 -n ray
```
删除pod
```commandline
kubectl delete pod automl
```

批量删除pod
```commandline
kubectl  -n ray get pods | grep Error | awk '{print$1}'| xargs kubectl delete pods

kubectl  -n ray get pods | grep job1 | awk '{print$1}'| xargs kubectl delete pods
 
kubectl  get pods -n kube-system | grep Evicted | awk '{print$1}'| xargs kubectl delete -n kube-system pods

```

Scaling Applications on Kubernetes with Ray
https://vishnudeva.medium.com/scaling-applications-on-kubernetes-with-ray-23692eb2e6f0  

##### Ray Job Submission
提交任务
```commandline
export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir":"./"}' -- python myscript.py

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 20 --task classification --metric log_loss --estimator_list '["lgbm"]' --seed 1

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric accuracy --estimator_list '["lgbm"]' --estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":3}' --seed 1

```

### k8s 提交job
```commandline
kubectl run ng2 --image=nginx --namespace=test --overrides='{ "apiVersion": "apps/v1", "spec": { "serviceAccount": "svcacct1" , "serviceAccountName": "svcacct1" }  }' -o yaml
kubectl -n ${ray_namespace} run ${ray_cluster_name}-job3 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --env="RAY_ADDRESS=${ray_cluster_name}-ray-head:8265" --overrides='{ "spec": { "restartPolicy": "Never", "serviceAccountName": "ray-user" }  }' -- /bin/bash -c "kubectl -n  ${ray_namespace} delete raycluster ${ray_cluster_name}  && helm -n ${ray_namespace} uninstall ${ray_cluster_name}" -o yaml

```

```commandline
kubectl -n ray run automl --image=fanshuangxi/automl:latest --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- python /home/submarine/automl_flaml_interface/automl_main.py --task_id test --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 240 --task classification --metric roc_auc --estimator_list "['lgbm','xgboost']"  --seed 1

kubectl -n ray run automl-test-job1 --image=fanshuangxi/automl:flaml-0.1.4 --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- bash ./entrypoint.sh --task_id test --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 240 --task classification --metric roc_auc --estimator_list "['lgbm','xgboost']"  --seed 1
kubectl -n ray run automl-test-job2 --image=fanshuangxi/automl:flaml-0.1.4 --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- bash ./entrypoint.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 60 --task classification --metric roc_auc --estimator_list '["lgbm"]' --seed 1

# submmit with multiple commands
# bash -c "command"
kubectl -n ray run automl-job1 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --overrides='{ "spec": { "restartPolicy": "Never" }' --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- /bin/bash -c "git clone xxx && cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric log_loss --estimator_list '[\"lgbm\"]' --estimator_kwargs '{\"n_jobs\":4,\"n_concurrent_trials\":2}' --seed 1"

kubectl -n ray run automl-job2 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --overrides='{ "spec": { "restartPolicy": "Never", "containers": [ { "name":"ray", "image":"fanshuangxi/automl:ray-flaml-cpu-0.1.3", "imagePullPolicy": "Always", "env": [{"name": "HADOOP_USER_NAME","value": "prod_alita"}, {"name": "HADOOP_USER_PASSWORD","value": "xxx"}], "command": ["/bin/bash", "-c", "--"], "args": [ "git clone xxx && cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric log_loss --estimator_list '[\"lgbm\"]' --estimator_kwargs '{\"n_jobs\":4,\"n_concurrent_trials\":2}' --seed 1 " ], "resources": { "requests": { "cpu": 4, "memory": "16Gi" } } } ] } }'
# 转义
kubectl -n ray run automl-job2 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.3 --overrides="{ \"spec\": { \"restartPolicy\": \"Never\", \"containers\": [ { \"name\":\"ray\", \"image\":\"fanshuangxi/automl:ray-flaml-cpu-0.1.3\", \"imagePullPolicy\": \"Always\", \"env\": [{\"name\": \"HADOOP_USER_NAME\",\"value\": \"prod_alita\"}, {\"name\": \"HADOOP_USER_PASSWORD\",\"value\": \"xxx\"}], \"command\": [\"/bin/bash\", \"-c\", \"--\"], \"args\": [ \"git clone xxx && cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric log_loss --estimator_list '[\\\"lgbm\\\"]' --estimator_kwargs '{\\\"n_jobs\\\":4,\\\"n_concurrent_trials\\\":2}' --seed 1 \" ], \"resources\": { \"requests\": { \"cpu\": 4, \"memory\": \"16Gi\" } } } ] } }"

kubectl -n ray run automl-job4 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.2 --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- /bin/bash -c "git clone http://didi-automl:xxx@git.xiaojukeji.com/dml/dml-autotabular && cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 20 --task classification --metric log_loss --estimator_list '[\"lgbm\"]' --seed 1"

kubectl -n ray run automl-job5 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.1-dev --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- /bin/bash -c  "cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 20 --task classification --metric log_loss --estimator_list '[\"lgbm\"]' --seed 1"
kubectl -n ray run automl-job6 --image=fanshuangxi/automl:ray-flaml-cpu-0.1.1-dev --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=xxx" -- /bin/bash -c  "cd dml-autotabular && bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric log_loss --estimator_list '[\"lgbm\"]' --estimator_kwargs '{\"n_jobs\":4,\"n_concurrent_trials\":1}' --seed 1"

```

### 通过k8s向Ray cluster 提交任务
向ray cluster 提交任务：
可通过ymal文件提交
kubectl port-forward 通过端口转发映射本地端口到指定的应用端口，从而访问集群中的应用程序(Pod).

kubectl run 直接提交ray cluster 任务报错：
Error from server (Forbidden): services "automl-cluster1-ray-head" is forbidden: User "system:serviceaccount:ray:default" cannot get resource "services" in API group "" in the namespace "ray"

#### k8s集群内部不需要设置port-forward
service/automl-cluster1-ray-head 8265 设置会报错
kubectl -n ray port-forward service/automl-cluster1-ray-head 8265:8265

```commandline
# 注意 k8s集群内部不需要设置port-forward   
kubectl -n ray run automl-cluster1-job2 --image=rayproject/ray:latest --env="RAY_ADDRESS=automl-cluster1-ray-head:8265" -- /bin/bash -c "git clone xxx && cd dml-autotabular && ray job submit --runtime-env-json='{\"working_dir\":\"./\", \"env_vars\": {\"HADOOP_USER_NAME\": \"prod_alita\", \"HADOOP_USER_PASSWORD\": \"xxx\"}}' -- bash ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric roc_auc --estimator_list '[\"lgbm\"]' --estimator_kwargs '{\"n_jobs\":4,\"n_concurrent_trials\":1}'  --seed 1"

```

通过 yml文件创建任务
```commandline
kubectl -nray create -f https://k8s.io/examples/application/shell-demo.yaml

```
git 添加下载权限
User Settings 添加Access Tokens
```commandline
# user  didi-automl:xxx
git clone http://didi-automl:xxx@git.xiaojukeji.com/dml/dml-autotabular
```

imagePullPolicy, 镜像的拉取策略
```commandline
Always       总是拉取镜像
IfNotPresent 本地有则使用本地镜像,不拉取
Never        只使用本地镜像，从不拉取，即使本地没有
如果省略imagePullPolicy  策略为always 
```

### 容器中环境变量不生效问题
```commandline
经验证： 
~/.bashrc 中设置的环境变量，进入容器中不生效，source ~/.bashrc 无效
/etc/profile 中设置的环境变量,进入容器中生效
```

另外，
可在dockerfile中设置环境变量
将CLASSPATH 的设置放入run.sh 脚本中
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

参考 
https://stackoverflow.com/questions/46834331/set-environment-variables-in-docker 

### 服务账户
### Role, ClusterRole, RoleBinding
https://blog.z0ukun.com/?p=312 

查询服务账户以及命名空间中的其他 ServiceAccount 资源
```commandline
kubectl get serviceAccounts -n ray

# 查询服务帐户对象的完整信息
kubectl get serviceaccounts/ray-user -o yaml
```
要使用非默认的服务账户，将 Pod 的 spec.serviceAccountName 字段设置为你想用的服务账户名称

#### serviceaccount 权限问题
```commandline
#Error: list: failed to list: secrets is forbidden: User "system:serviceaccount:default:default" cannot list resource "secrets" in API group "" in the namespace "default"
#Error: uninstall: Release not loaded: automl-cluster4: query: failed to query with labels: secrets is forbidden: User "system:serviceaccount:ray-d:ray-operator-serviceaccount" cannot list resource "secrets" in API group "" in the namespace "ray-d"
```
https://kubernetes.io/zh/docs/tasks/configure-pod-container/configure-service-account/
https://blog.csdn.net/u013189824/article/details/110232938
https://blog.csdn.net/u013189824/article/details/111047145
https://stackoverflow.com/questions/47973570/kubernetes-log-user-systemserviceaccountdefaultdefault-cannot-get-services 

https://blog.z0ukun.com/?p=312

A Step-by-Step Guide to Scaling Your First Python Application in the Cloud
https://medium.com/distributed-computing-with-ray/a-step-by-step-guide-to-scaling-your-first-python-application-in-the-cloud-8761fe331ef1 

### Ray on premise
https://docs.ray.io/en/latest/cluster/cloud.html 
https://medium.com/distributed-computing-with-ray/a-step-by-step-guide-to-scaling-your-first-python-application-in-the-cloud-8761fe331ef1

```commandline

# On the head node
ray start --head --port=6379

Next steps
  To connect to this Ray runtime from another node, run
    ray start --address='<ip address>:6379'
  Alternatively, use the following Python code:
    import ray
    ray.init(address='auto')
    
# connect to the cluster with 
Running a Ray program on the Ray cluster
To run a distributed Ray program, you’ll need to execute your program on the same machine as one of the nodes.

ray.init(address='auto')
        
  To connect to this Ray runtime from outside of the cluster, for example to
  connect to a remote cluster from your laptop directly, use the following
  Python code:
    import ray
    ray.init(address='ray://<head_node_ip_address>:10001')

# When you want to stop the Ray processes, run ray stop on each node.
ray stop


# ray start --head --node-ip-address 172.17.126.115 --redis-shard-ports=6379

提交任务
在head node上执行代码

export HADOOP_USER_NAME=prod_alita
export HADOOP_USER_PASSWORD=xxx
./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric roc_auc --estimator_list '["lgbm"]' --estimator_kwargs '{"n_jobs":-1,"n_concurrent_trials":3}'  --seed 1

```

方案：
通过任务参数判断是否为并行任务。
如果是并行，则初始化ray，否则不初始化ray
需要编写operator管理ray集群的启动，状态监控，删除等工作

该部分可参考ray_operator 接口，或使用配置文件管理

启动集群
先启动head ，然后启动每个worker 连接head
如果在head上执行程序，启动代码中添加
    import ray
    ray.init(address='auto')

如果在ray 集群外部执行程序，启动代码中添加
    import ray
    ray.init(address='ray://<head_node_ip_address>:10001')

在每个pod上查看 cpu，mem使用情况
通过观察，加载数据时，head上的内存上升较多，worker上的内存无变化，
可推断，加载数据只发生在head上，且该过程并未使用ray API所以没有做并行，
后续可考虑当并行执行时是否可使用 ray并行加载数据

资源提供方提供的pod是什么形式，是否带密码？


To see live updates to the status:
```commandline
# 不生效
$ watch -n 1 ray status
```


Cluster Launcher Commands
https://docs.ray.io/en/latest/cluster/commands.html#cluster-commands




