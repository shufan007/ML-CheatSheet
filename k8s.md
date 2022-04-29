K8s

https://kubernetes.io/zh/docs/home/ 

https://kubernetes.io/zh/docs/tutorials/kubernetes-basics/deploy-app/deploy-interactive/ 

https://kubernetes.io/zh/docs/tutorials/hello-minikube/ 

使用 kubeadm 创建集群
https://kubernetes.io/zh/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/ 

kubectl 备忘单
https://kubernetes.io/zh/docs/reference/kubectl/cheatsheet/ 

摆脱 AI 生产“小作坊”：如何基于 Kubernetes 构建云原生 AI 平台
https://zhuanlan.zhihu.com/p/504047864 

——————————
10.83.160.44

10.83.237.69

10.83.155.19

10.83.152.16 

10.83.161.15

~/.kube/config


—————————————

~/.kube/config


brew install kubectl 

kubectl version --client
kubectl cluster-info

kubectl get pods                              # 列出当前命名空间下的全部 Pods


brew install helm


Installation Ray Cluster

# Navigate to the directory containing the chart
$ cd ray/deploy/charts

# Install a small Ray cluster with the default configuration
# in a new namespace called "ray". Let's name the Helm release "example-cluster."
$ helm -n ray install example-cluster --create-namespace ./ray


kubectl describe pods kubectl -n ray get pods
kubectl describe pods ray-operator-5b985c9d77-wdt6d

查看pods
kubectl describe pods calico-node-9l2b4 -n kube-system

kubectl describe pods -n ray example-cluster-ray-worker-type-s46vm


https://hub.docker.com/r/rayproject/ray

kubectl logs  $(kubectl get pod -l cluster.ray.io/component=operator -o custom-columns=:metadata.name) | tail -n 100

查看pod
kubectl -n ray describe pod example-cluster-ray-head-type-w5pmd

进入容器
kubectl -n ray exec -it example-cluster-ray-head-type-w5pmd -c ray-node -- /bin/bash


删除job

kubectl get jobs -n ray

kubectl delete job ray-test-job1 -n ray



批量删除pod
kubectl  -n ray get pods | grep Error | awk '{print$1}'| xargs kubectl delete pods

kubectl  -n ray get pods | grep job1 | awk '{print$1}'| xargs kubectl delete pods
 
kubectl  get pods -n kube-system | grep Evicted | awk '{print$1}'| xargs kubectl delete -n kube-system pods


————————


You can use helm upgrade to adjust the fields minWorkers and maxWorkers without restarting the Ray cluster.

sudo helm -n ray upgrade example-cluster ./ray



Ray Job Submission
提交任务

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir":"./"}' -- python myscript.py

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 20 --task classification --metric log_loss --estimator_list '["lgbm"]' --seed 1

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data ./data/iris_data.csv --label_col label --output_path ./test.outputs --time_budget 60 --task classification --metric accuracy --estimator_list '["lgbm"]' --estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":3}' --seed 1


export HADOOP_USER_NAME=prod_alita
export HADOOP_USER_PASSWORD=JEnvRCY0t7vxys5GE4TEcSNqRHn2VHRN

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 240 --task classification --metric roc_auc --estimator_list '["lgbm"]' --estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":5}'  --seed 1


k8s 提交job

kubectl run automl --image=fanshuangxi/automl:latest --env="HADOOP_USER_NAME=prod_alita" --env="HADOOP_USER_PASSWORD=JEnvRCY0t7vxys5GE4TEcSNqRHn2VHRN" -- python /home/submarine/automl_flaml_interface/automl_main.py --task_id test --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --time_budget 240 --task classification --metric roc_auc --estimator_list "['lgbm','xgboost']"  --seed 1


