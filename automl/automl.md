# AutoML:

https://zhuanlan.zhihu.com/c_1165403884117299200 

AutoML框架概览
https://zhuanlan.zhihu.com/p/378722852 

## AutoML 理论
走马观花AutoML
https://zhuanlan.zhihu.com/p/212512984 

## FLAML 

https://zhuanlan.zhihu.com/p/410362009 

https://www.kdnuggets.com/2021/09/fast-automl-flaml-ray-tune.html

https://zhuanlan.zhihu.com/p/405177840 


## Ray

理论
论文：
https://arxiv.org/abs/2005.01571 
https://arxiv.org/pdf/1911.04706.pdf 

Blendsearch-on-Ray-benchmark

Ray/Ray Tune：
https://docs.ray.io/en/latest/index.html

https://docs.ray.io/en/master/tune/index.html 

Large Scale Training at BAIR with Ray Tune
https://bair.berkeley.edu/blog/2020/01/16/tune/ 

Ray.tune
https://zhuanlan.zhihu.com/p/93643769 

Ray配置
https://docs.ray.io/en/latest/ray-core/configure.html#configuring-ray 

Deploying on Kubernetes
https://docs.ray.io/en/master/cluster/kubernetes.html#ray-k8s-deploy 

强化学习框架 Ray 在 K8s 上的自动伸缩设计与实现
https://cloud.tencent.com/developer/news/463783 

Ray.tune：把超参放心交给它
https://zhuanlan.zhihu.com/p/93643769 


Ray基本使用方法(draft)
https://zhuanlan.zhihu.com/p/102557115 

分布式框架Ray及RLlib简易理解
https://zhuanlan.zhihu.com/p/61818897

Ray Tune Hyperparameter Optimization Framework
https://blog.csdn.net/u011254180/article/details/81178651 


超参优化工具总结(4)——Ray.tune
https://zhuanlan.zhihu.com/p/93584289 

PyTorch + Ray Tune 调参
https://blog.csdn.net/tszupup/article/details/112059788 


Pytorch 分布式训练
https://zhuanlan.zhihu.com/p/76638962 

Pytorch多机多卡分布式训练
https://zhuanlan.zhihu.com/p/68717029 

Ray Cluster Launcher: a utility for managing resource provisioning and cluster configurations across AWS, GCP, and Kubernetes.

Frameworks: Ray Train is built to abstract away the coordination/configuration setup of distributed deep learning frameworks such as Pytorch Distributed and Tensorflow Distributed, allowing users to only focus on implementing training logic.

从MR到Spark再到Ray，谈分布式编程的发展
https://zhuanlan.zhihu.com/p/98033020 

Ray的系统层是以Task为抽象粒度的，用户可以在代码里任意生成和组合task，比如拆分成多个Stage,每个Task执行什么逻辑，每个task需要多少资源,非常自由，对资源把控力很强。RDD则是以数据作为抽象对象的，你关心的应该是数据如何处理，而不是去如何拆解任务，关心资源如何被分配，这其中涉及的概念比如Job,Stage,task你最好都不要管，RDD自己来决定。虽然抽象不一样，但是最终都能以此来构建解决各种问题的应用框架，比如RDD也能完成流，批，机器学习相关的工作，Ray也是如此。所以从这个角度来看，他们其实并没有太过本质的区别。从交互语言而言，双方目前都支持使用python/java来写任务。另外，就易用性而言，双方差距很小，各有优势。

现在Ray来了，Ray吸取了Spark的诸多营养，走的更远了，首先，既然Python作为世界上当前最优秀的人机交互通用语言，我直接就让Python成为头等公民，安装部署啥的全部采用Python生态。比如你pip 安装了ray,接着就可以用ray这个命令部署集群跑在K8s/Yarn或者多个主机上，也可以直接使用python引入ray API进行编程。易用性再次提高一大截。其次，作为分布式应用，他除了能够把python函数发到任意N台机器上执行，还提供了Actor(类)的机制，让你的类也可以发到任意N台机器上执行。


开源史海钩沉系列 [1] Ray：分布式计算框架
https://zhuanlan.zhihu.com/p/104022670


### Katib

云原生的自动机器学习系统 Katib 论文解读
https://zhuanlan.zhihu.com/p/157589799 

首先是多租户，Katib 是目前开源的超参数搜索系统里唯一原生支持多租户的。其次是分布式的训练能力，Katib 构建于 Kubeflow 的众多项目之上，能够支持分布式的模型训练与并行的超参数搜索。然后是云原生，Katib 是一个 Kubernetes Native 的系统，所有的功能都依托于 Kubernetes 的扩展性能力实现，是一个云原生的系统。最后，是扩展性，Katib 的架构易于扩展，能够集成不同的超参数搜索算法。

Kubeflow机器学习工具包-概述
https://blog.csdn.net/chenxy02/article/details/123426242 


## spark grid search和random search
开发计划及技术方案：
1. 考察验证 spark grid search和random search方案。
参数网格构建方法：
grid search：
可直接使用 spark.ml.tuning 中的ParamGridBuilder创建参数网格。
参数网格构建示例：
val paramGrid = new ParamGridBuilder()
  .addGrid(booster.maxDepth, Array(3, 8))
  .addGrid(booster.eta, Array(0.2, 0.6))
  .build()

random search：
因spark.ml中没有直接提供构建随机参数参数的方法，需要实现一个类似ParamGridBuilder的构造器，进行random参数网格的构造。
经考察，可借助breeze.stats.distributions中提供的分布函数组件实现一个生成随机参数网格构造器RandomGridBuilder。
随机参数网格构建示例：
val randomGrid = new RandomGridBuilderDebug(10)
  .addDistr(lgbm.learningRate,Uniform(0.05,0.2))
  .addDistr(lgbm.maxBin,Uniform(1,2).map(x=>scala.math.round(x)))
  .build()
 
模型选择验证方法：
可使用spark.ml.tuning中的CrossValidator 和 TrainValidationSplit。
CrossValidator 做交叉验证，计算开销大，结果相对稳定。
TrainValidationSplit 直接做训练集和验证集的切分，计算开销小。
 
Cross Validator 构建方法如下：
val cv = new CrossValidator()
  .setEstimator(trainer)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
Train Validation Split构建方法如下：
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(trainer)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.8)

1. spark-ml-packages 工程代码学习，走通grid search验证流程。
熟悉spark-ml-packages 工程代码；
调通grid search验证流程。

1. 产品功能和接口方案设计：

功能：
验证方法：支持cross_validator和train_valid_split，后续扩展支持用户自定义验证集。
调参方法：支持grid_search, random_search， 后续可考虑增加高斯过程（贝叶斯调参）方法。
模型：支持xgboost，lightgbm，后续扩展支持 TreeLift模型。
模型任务：支持二分类、多分类、回归。
 
超参调优的接口：
调研主流产品automl超参调优的接口，设计；
添加参数：
validatorMethod: 模型选择
* cross_validator:  nFolds：交叉验证折数 >=2，<=10, 默认3
* train_valid_split：trainRatio: 训练集比例（0，1 ），默认0.7
turning_method: grid_search, random_search
 
模型评估：添加metric，  setMetricName
 
拟开放可搜索参数：
 
分类：

xgboost：
eta	学习率
maxDepth	树最大深度
subsample	数据采样比例
lambda	L2正则系数
alpha	L1正则系数

lightgbm：
numIterations	迭代次数	100
learningRate	学习率	0.1
maxDepth	树最大深度	-1
numLeaves	叶子节点数	31
 
回归：
 
待补充
 
1. 开发计划
 
第一步：

基于xgboost
先走通二分类 
grid search
cv， 
split，
添加metric

random search

第二步： 多分类
第三步：回归

第四步：xgboost 版本 代码重构，扩展至lightgbm

第五步：合并xgboost， lightgbm 接口

第6步： 集成进 Lift 的功能。（可考虑放到第四步之后）
 
 
xgb、lightgbm保存的native模型改造
将spark 的 xgb、lightgbm保存的native模型，可使用python版（python单机版xgb，lightgbm模型载入）解析
 
XGB：
bestModel.nativeBooster.saveModel(nativeModelPath)
 
参考官方文档：
https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html 
 
val nativeModelPath = "/tmp/nativeModel"
xgbClassificationModel.nativeBooster.saveModel(nativeModelPath)
 
import xgboost as xgb
bst = xgb.Booster({'nthread': 4})
bst.load_model(nativeModelPath)
 
测试该方法保存模型的有效性
注意：输入数据格式如何定义
验证：搭建python环境（安装xgb，lightgbm）。测试spark保存的native model 在python上载入，预测。注意输入数据格式。
 
预测准确性问题：特征列没有对齐，下标问题？模型保存时如何保存特征列？
 
GBM：
支持模型保存，并且保存后的模型和Python等语言是可以相互调用的。
参考：
https://jishuin.proginn.com/p/763bfbd60633 
验证可行性：
与 python环境测试
当前验证保存为多文件，非单文件，跟上面参考文档中不一致。无法在python中直接读取。读取里面的txt文件后，预测结果差异很大，需要研究原因。载入的模型，如何获取参数。数据对齐问题。

——————————

TODO：
Summary： 搜索中间状态打日志
评估指标： 与现有保持一致，枚举之解析问题
外部提供验证集：需要考察，参考train_valid_split 方案开发接口。
参数配置文档： 跟前后端沟通
添加UPLift功能

————————

计划：

1. 基于spark的 automl tuning 组件开发，部署上线。
指标：
1）核心功能覆盖：
验证方法：支持cross_validator，train_valid_split，用户自定义验证集三种方式。
调参方法：支持grid_search, random_search。
支持算法：lightgbm，xgboost 。
调参超参数支持：学习率，树最大深度，迭代次数，最大分箱数，L1，L2正则系数。
支持任务类型：分类， 回归。
评估方法：支持常用评估方法。
BinaryClassification：(areaUnderROC|areaUnderPR)
MulticlassClassification：(f1|weightedPrecision|weightedRecall|accuracy)
Regression：(mse|rmse|r2|mae)

指标：
2）效果：与经验值参数相比评价指标提升>2%
3）与手动调参相比评价指标提升>1%

2. spark xgboost、lightgbm保存native模型，实现python api可解析模型。

指标：
1）完成功能开发，上线： spark保存及载入native模型，inference 端解析native模型。
2）模型准确性验证一致。spark保存模型，python读取验证，inference端解析验证。

3. AutoML模型自动化搜索技术。

指标：
1）完成技术调研与方案选型；
2）打通基于新框架的模型训练验证流程；

性能：
3）比grid_search/random_search调参性能提升>10倍 
4）1小时训练时间的下的效果提升>2%


4.合作场景&技术沉淀影响：

指标：
1）合作场景数 >=2
2）技术文档 >=2

———————————————————————
组件开发部署， 问国灏部署流程

基于FLAML的智能autuml 算法 研究，验证流程打通。

模型部署平台闭环

调参性能提升
写周报：


接手spark 传统机器学习组件


跟松波对接Geo数据训练，数据规模：350万行* 550列，
数据拆分时报错，经调整多组spark参数仍不能跑通，后改用sql组件拆分数据
使用XGBoost训练

使用lightgbm训练时常出现OOM或资源超限报错，难以调出可正常运行的参数。
运行时间较长（单次训练>=30分钟），增大execute数量对性能提升不明显。

Geo数据 XGBoost 调参执行比较耗时

dml-engine 开发环境配置

使用lightgbm automl 超参搜索，报错
org.apache.spark.SparkException: Could not find CoarseGrainedScheduler.

不管申请多少executor，实际申请到的只有15个

研究 Lightgbm  spark并行策略

跟松波又做了沟通，重新生成样本，统一数据处理标准，数据需要重新申请权限。

尝试dataframe repartition
————————————————————————————
计划：

梳理xgboost_component 代码逻辑，测试代码逻辑，可执行原测试流程。
基于xgboost_component 另写一份native model模型加载代码
编写基于native model的测试代码，重新生成native model，走通测试流程。

如何从nativemodel中得到num_class?
查看xgboost c++ 源码
查看python model load代码
如何从xgboost native model中区分是分类还是回归？

参考Spark 并行预测解决方案文档：https://toutiao.io/posts/5wgxivz/preview 


TODO：
spark 跑模型

spark 大数据量跑模型问题排查:

为了防止block，将训练数据分区数和申请的工作节点数numWorkers保持一致。即对训练数据做repartition处理。numWorkers默认值是32.
 经验3:numWorker参数应该与executor数量设置一致，executor-cores设置小一些
 【来源：https://python.iitter.com/other/280790.html，转载请注明】

继续排查 HashMap的问题
cannot assign instance of scala.collection.mutable.HashMap to field com.microsoft.ml.spark.lightgbm.LightGBMSummary.com$microsoft$ml$spark$lightgbm$LightGBMSummary$$trainRecords of type scala.collection.mutable.LinkedHashMap in instance of com.microsoft.ml.spark.lightgbm.LightGBMSummary
	at java.io.ObjectStreamClass$FieldReflector.setObjFieldValues(ObjectStreamClass.java:2133)

Geo 模型调参：
可先用xgboost训练，调研如何添加mape metric 或者udf metric
尝试对训练数据repartition

可找松波了解数据处理的过程，松波设置了executor =500， spark 参数？
研究：

val numWorkers = SparkUtils.calcNumWorkers(trainDF)

.setNumWorkers(numWorkers)

lightgbm：
 setNumThreads


XGBoost Native model:
为什么python save_model 保存的模型，在c++ 中无法加载？
而xgboost4j中改造后的模型保存native模型之后 可以加载。

查看 XGBoosterLoadModel 源码


MAPE 实现的差异：
sklearn ：

epsilon = np.finfo(np.float64).eps
mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)

lightgbm  bojective 中实现：

label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i]));

const double diff = score[i] - label_[i];
gradients[i] = static_cast<score_t>(Common::Sign(diff) * label_weight_[i]);

—————


Inference: 
c++, java, go

——————————

class_num 输出到matadata文件，兼容原来模型，通过 num_class 参数区分。 验证

Geo  xgboost 优化 ，向松波要参数，超参调优 。

xgboost， lightgbm spark 参数优化。 数据repartation。 下周
———————————


Pip 安装 指定源

 -i https://pypi.tuna.tsinghua.edu.cn/simple

 阿里云 http://mirrors.aliyun.com/pypi/simple/ 
 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/ 
 豆瓣(douban) http://pypi.douban.com/simple/ 
 清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/ 
 中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

——————————————————
AutoML 框架：
AutoML框架概览：https://zhuanlan.zhihu.com/p/378722852 


FLAML： https://zhuanlan.zhihu.com/p/410362009 
Ray Tune 
NNI
Kubeflow/Katib：https://zhuanlan.zhihu.com/p/157589799 
https://zhuanlan.zhihu.com/p/77760872 

FLAML + Ray Tune： https://www.kdnuggets.com/2021/09/fast-automl-flaml-ray-tune.html 


TOD：
基于luban 测试：现有数据集验证 geo
Nni：特征选择 验证 ，对接，集成


--tableNames "{\"train\":\"alita_dev.ml_self_sql_14760305_02330823_07b63612d4_1\",\"validate\":\"alita_dev.ml_self_sql_14760305_54285886_2d32902be6_1\"}" \


验证pyarrow 读hdfs

TODO：
完成flaml jupyter 镜像
验证spark lgbm 模型训练效果差的原因
继续测试flaml criteo模型调参效果：数据整理
测试flaml 模型搜索效果

automl接口设计
执行命令提交，模型保存问题

flaml 并行训练研究

3、automl作为创建实验的一种选项，替代原有的predefined template集成那个；@樊双喜 Shuangxi Fan


# use lgbm only
settings = {
    "time_budget": 3600,  # total running time in seconds
    "metric": 'log_loss',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                           # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": 'classification',  # task type
    "estimator_list": ['lgbm'],
    #"estimator_list":['lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree'],
    "categorical_feature": categorical_feature,
    #"n_jobs": 45,
    "log_file_name": 'criteo_experiment_1.log',  # flaml log file
    "seed": 1,    # random seed
}


'''The main flaml automl API'''
automl.fit(X_train=X_train, y_train=y_train, **settings)


task_id：automl和前端对接

# 数据参数
train_data: hdfs 路径
test_data：hdfs 路径
label_col：标签列

# 模型搜索参数
time_budget：int 单位：秒
task： 任务类型

estimator_list：['lgbm', 'xgboost',  'catboost'],
metric：
"seed": 1


输出：
执行日志文件
最优模型参数
模型文件

classification

AutoX
—————
submarine automl：
TO DO：
flaml native模型保存

模型下载：
方案1，将模型保存在挂载目录中，通过notebook下载
2.链接数梦，模型部署，下载

分布式部署方案调研

AutoML 

Fast AutoML with FLAML + Ray Tune

https://www.kdnuggets.com/2021/09/fast-automl-flaml-ray-tune.html 


FLAML is a newly released library containing state-of-the-art hyperparameter optimization algorithms. FLAML leverages the structure of the search space to optimize for both cost and model performance simultaneously. It contains two new methods developed by Microsoft Research:
* Cost-Frugal Optimization (CFO)
* BlendSearch
Cost-Frugal Optimization (CFO) is a method that conducts its search process in a cost-aware fashion. The search method starts from a low-cost initial point and gradually moves towards a higher cost region while optimizing the given objective (like model loss or accuracy).
Blendsearch is an extension of CFO that combines the frugality of CFO and the exploration ability of Bayesian optimization. Like CFO, BlendSearch requires a low-cost initial point as input if such point exists, and starts the search from there. However, unlike CFO, BlendSearch will not wait for the local search to fully converge before trying new start points.

How CFO and BlendSearch Work
CFO begins with a low-cost initial point (specified through low_cost_init_value in the search space) and performs local updates following its randomized local search strategy. With such a strategy, CFO can quickly move toward the low-loss region, showing a good convergence property. Additionally, CFO tends to avoid exploring the high-cost region until necessary. This search strategy is further grounded with a provable convergence rate and bounded cost in expectation.

BlendSearch further combines this local search strategy used in CFO with global search. It leverages the frugality of CFO and the space exploration capability of global search methods such as Bayesian optimization. Specifically, BlendSearch maintains one global search model, and gradually creates local search threads over time based on the hyperparameter configurations proposed by the global model. It further prioritizes the global search thread and multiple local search threads depending on their real-time performance and cost. It can further improve the efficiency of CFO in tasks with complicated search space, e.g., a search space that contains multiple disjoint, non-continuous subspaces.

How to scale up CFO and BlendSearch with Ray Tune’s distributed tuning
  To speed up hyperparameter optimization, you may want to parallelize your hyperparameter search. For example, BlendSearch is able to work well in a parallel setting: It leverages multiple search threads that can be independently executed without obvious degradation of performance. This desirable property is not always true for existing optimization algorithms such as Bayesian Optimization.
To achieve parallelization, FLAML is integrated with Ray Tune. Ray Tune is a Python library that accelerates hyperparameter tuning by allowing you to leverage cutting edge optimization algorithms at scale. Ray Tune also allows you to scale out hyperparameter search from your laptop to a cluster without changing your code. You can either use Ray Tune in FLAML or run the hyperparameter search methods from FLAML in Ray Tune to parallelize your search. The following code example shows the former usage, which is achieved by simply configuring the n_concurrent_trials argument in FLAML.

Conclusion

FLAML is a newly released library containing state-of-the-art hyperparameter optimization algorithms that leverages the structure of the search space to optimize for both cost and model performance simultaneously. FLAML can also utilize Ray Tune for distributed hyperparameter tuning to scale up these economical AutoML methods across a cluster.

AutoML 理论
走马观花AutoML
https://zhuanlan.zhihu.com/p/212512984 

FLAML 理论
论文：
https://arxiv.org/abs/2005.01571 

Blendsearch-on-Ray-benchmark

————————————————

Ray Tune：
https://docs.ray.io/en/master/tune/index.html 


You can leverage your Kubernetes cluster as a substrate for execution of distributed Ray programs. The Ray Autoscaler spins up and deletes Kubernetes Pods according to the resource demands of the Ray workload. Each Ray node runs in its own Kubernetes Pod.


# image is Ray image to use for the head and workers of this Ray cluster.
# It's recommended to build custom dependencies for your workload into this image,
# taking one of the offical `rayproject/ray` images as base.
image: rayproject/ray:latest


Ray on k8s

方案一：

Ray集群资源固定配置
资源配置：CPU，GPU，memory
先在k8s上拉起Ray集群
task提交到Ray集群，由Ray管理

RayCluster image 使用自定义image，
安装flaml 组件
安装hdfs 链接工具和配置
 
image: rayproject/ray:latest

operatorImage 使用官方镜像 `rayproject/ray`
operatorImage: rayproject/ray:latest

Ray provides a Helm chart to simplify deployment of the Ray Operator and Ray clusters. 

Using Ray Client to connect from within the Kubernetes cluster

提交任务：
使用k8s起一个提交任务的容器，连接Ray Cluster
镜像使用官方镜像: rayproject/ray:latest
提交任务的entrypoint 代码从github下载


当n_concurrent_trials 设置超过可满足n_jobs 的worker数时，实际并行数为满足n_jobs 的worker数。
当不走并行调度（设置n_concurrent_trials=1，或不设置）时，向Ray集群提交任务，默认在head node上执行。
考虑：当head资源不足时会怎么样？是否会调度到worker上？
	head 的调度是否可以关闭？


TODO：
一个Ray worker node上的多个workers是指什么？跟n_jobs的关系？
内存的问题：验证并行任务时看到head上的内存占用比较高，大数据的并行策略如何？能否数据并行？

当没有任务执行的时候，Ray集群是否一直占用k8s资源？
 看到没有任务时，Ray node上的workers 是0

资源配置的细节问题
并行的原理和效果
GPU如何配置，如何使用


方案二：

对于大任务可以使用该方案，需判断任务大小，根据参数或数据大小判断
需管理任务，校验任务状态

根据提交的任务，配置资源
拉起相应资源的Ray集群
然后把任务提交到Ray集群
任务完成后删除Ray集群，释放资源

方案三：

直接基于k8s 部署automl
需要管理并行训练的任务，可基于kubeflow/katib，但上层应用不太丰富。

对于小任务，可以直接在k8s上提交单机训练任务。


问题：

为什么开启并行trails 可以防止OOM？


Running Ray programs with Ray Jobs Submission


export HADOOP_USER_NAME=prod_alita
export HADOOP_USER_PASSWORD=JEnvRCY0t7vxys5GE4TEcSNqRHn2VHRN

ray job submit --runtime-env-json='{"working_dir":"./"}' -- sh ./run.sh --task_id task_123456 --train_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_1/part-00000-a81f8fff-c57a-42ce-9afc-073cd801e9e6-c000.snappy.parquet --test_data hdfs://DClusterNmg2/user/prod_alita/alita_dev/hive/alita_dev/ml_data_split_13434844_59261508_49e97caa66_2/part-00000-55dfe73a-092f-44f2-92b0-e2801b5ad261-c000.snappy.parquet --label_col label --output_path ./test.outputs --time_budget 240 --task classification --metric roc_auc --estimator_list '["lgbm"]' --estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":5}'  --seed 1
 

