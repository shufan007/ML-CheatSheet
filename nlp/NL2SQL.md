NL2SQL/TableQA

介绍文档
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650832341&idx=4&sn=54745ffbf8df8116b2fe7736d0f12e2d&chksm=84e5b9abb39230bd4cee833e34e7e61e7b4317d0f2c5972fd59aca38c8730336802f9c5c6c8d&scene=21#wechat_redirect 

https://www.jiqizhixin.com/articles/2022-01-23 

https://www.jiqizhixin.com/articles/2019-12-27-11 

NL2SQL 赛题说明
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650764672&idx=2&sn=bb8d8bb191bfb4b4093e874193019724&scene=21#wechat_redirect 

基于Bert的NL2SQL模型：一个简明的Baseline
https://kexue.fm/archives/6771 

https://blog.csdn.net/qq_16949707/article/details/96387107 

中文 NL2SQL 的冠军解决方案 https://zhuanlan.zhihu.com/p/86748426 

PPT: https://github.com/nudtnlp/tianchi-nl2sql-top1 

PLUG模型在线服务 2.0
https://nlp.aliyun.com/portal#/alice 

AliceMind项目地址：https://github.com/alibaba/AliceMind
https://gitee.com/mirrors_alibaba/AliceMind 
AliceMind/SDCUP.  table understanding

智能小Q:
https://help.aliyun.com/document_detail/140992.html 

数据集介绍
https://github.com/yechens/NL2SQL 

Chinese Word Vectors 中文词向量

https://github.com/Embedding/Chinese-Word-Vectors 
——————————————————————————
问题整理：

训练样本获取：

考查如何学习元模型，迁移学习，实体映射/对新表的关键词进行映射。
One-shot/feal-shot方法

如何构建数据集，调研
1.使用现有基于规则方式的问题和sql对
2.填空方式生成问题，模型生成sql，人工校准
生成问题时可考虑从当前数据集中替换实体生成问题

问题：

训练集中有data base id / table，预测的时候如何得到data base id/ table的信息？
首先从问题中推断出表，
预测的时候不需要输入表结构信息？表结构信息已表达进模型中？
同样：训练模型学习所学习的链接关系应该主要是表达逻辑，而不包含表字段映射。
表字段映射仍需要对特定任务进行训练。
如何解决训练和预测的输入不一致问题？ 还是所有的table schema放一起送人模型

训练时输入：问题+表头字段
产品化应用时，需要先预测表，然后将预测的表的表头输入模型，两阶段
多表情况下需考虑区分的问题。
两阶段合并一阶段：

意图识别：任务，表识别
条件值：条件值并不总是question的一个片段，条件值的格式化，校准

如果模型训练时需要与表绑定，那预训练模型训练训练的是什么能力？如何解除与表的绑定？除表以外的链接能力和逻辑组合能力？多表如何处理？

条件值匹配问题：
首先非常重要的就是数据预训练，我们需要统一问句与标注中的不同表现形式，例如需要根据问句中的"5 百元"定位到表格中的"500"。研究者会使用规则修正数字、日期、年份、单位；使用编辑距离修正同义表达；这两者都旨在统一 Query 为确定的格式。

表格数据量很大的情况，如何高效的对值进行学习。
只对已加工过的数据资产表搜索，不对大数据量的原始表

其实模型比较难解决的还是 WHERE 子句，我们发现该子句中确定特征列，及确定特征列的值是最困难的，它们的准确率相对其它子任务都要低一些。


追一科技表示本次竞赛数据集与 WikiSQL 主要有三点差异/难点，它们也是本次比赛中选手需要去解决的核心问题。
1. 表格内容对选手是可见的，并且问题具有泛化性 ，「腾讯控股的股票今天涨了多少」VS 鹅厂今天涨了多少」 
2. 省略显式的列名信息，更口语化的表达 
3. 数字的单位问题
 模型的处理流程和依赖关系都可以改进，因此可以总结为如下三个方向：
1. BERT、XLNet 等更好的输入特征 
2. 更合理的模型结构：目前的基线模型是 pipeline 的方式，预测完 condition column 之后再去预测 condition op 和 condition value。但在真实数据中，我们需要一种联合预测的方式来解决问题。 
3. 如何合理地将表格内数据也结合进模型，以前的方案都忽略了表格内数据。 

delta：https://github.com/Delta-ML/delta 

如果delta本身只是从页面查询变成自然语言问答，那么是比较有限的哈。如果能挖出点，上下文交互，基于数据洞见更智能的东西还是可以的

我们本身现有工具体量已经有了，往下更希望dia能承接发现数据价值的作用

DIA 只要保持目前 dc 应答交互的方式，那么自然语言的理解和低成本的查询那就会是它的所需，否则那么多做同类产品的商业公司直接配置些规则好了，不必那么折腾。

数据洞见的是一个通用能力，不是 DIA 所特有需要的，是在整个分析产品建设中要提供的公共基座。数易、异动分析等都是核心应用场景和沉淀，DIA 会参与能力建设，但也是一个能力的渠道出口。

目前能看到围绕 DIA 产品形态真正特有的能力是 NLP 这块，技术有限投入的情况下我们还是希望做一定的探索。

嗯嗯，是的，dia的自然语言交互是比较确定的，洞察的话我觉得可以先通过预先定义好rule的情况下去做智能查询结果的分析或者问题推荐，比如说gmv下降了，那可以给出城市订单量的分布，看看哪个城市的订单量下降了，然后再给出这个城市的相关信息，或者一个城市的订单与我们算法预估的不一致，那会给出订阅信息，洞察可以先从几个点入手，给人一种这个确实很智能的感觉



——————————————————————————

这里展示几篇上面3个方向的论文，更多顶尖模型可以在 WikiSQL 基准中找到：
[1]X-SQL: reinforce context into schema representation, 2019 [3]Semantic Parsing with Syntax- and Table-Aware SQL Generation, 2018 [1]A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization, 2019
* WikiSQL 基准：https://github.com/salesforce/WikiSQL   ——————————————————


Text-to-SQL

TableQA 

Dataset:
SQUALL
Spider 
单表单轮 WikiSQL[2]
在深度学习端到端解决方案流行之前，这一领域的解决方案主要是通过高质量的语法树和词典来构建语义解析器，再将自然语言语句转化为相应的SQL。
作者：追一科技
链接：https://www.zhihu.com/question/323109035/answer/683975497
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

现在的解决方案则主要是端到端与SQL特征规则相结合。
以在WikiSQL数据集上的SOTA模型SQLova为例：首先使用BERT对Question和SQL表格进行编码和特征提取，然后根据数据集中SQL语句的句法特征，将预测生成SQL语句的任务解耦为6个子任务，分别是Select-Column、Select-Aggregation、Where-Number、Where-Column、Where-Operation以及Where-Value，不同子任务之间存在一定的依赖关系，最终使用提取到的特征，依次进行6个任务的预测。
￼


	WikiSQL数据集虽然是目前规模最大的有监督数据集，但其数据形式和难度过于简单：对于SQL语句，条件的表达只支持最基础的>、<、=，条件之间的关系只有and，不支持聚组、排序、嵌套等其它众多常用的SQL语法，不需要联合多表查询答案，真实答案所在表格已知等诸多问题的简化，所以在这个数据集上，SQL执行结果的准确率目前已经达到了91.8%。
	但同时存在一个问题，这样的数据集并不符合真实的应用场景。在真实的场景中，用户问题中的值非常可能不是数据表中所出现的，需要一定的泛化才可以匹配到；真实的表之间存在错综复杂的键关联关系，想要得到真实答案，通常需要联合多张表进行查询；每一张都有不同的意义，并且每张表中列的意义也都不同，甚至可能相同名字的列，在不同的表格中所代表的含义是不同的；真实场景中，用户的问题表达会很丰富，会使用各种各样的条件来筛选数据。诸如此类的实际因素还有很多。因此，WikiSQL数据集起到的作用更多程度上是抛砖引玉，而不具备实际应用场景落地的价值。
	相比之下，Spider等数据集更贴近于真实应用场景：涉及到查询语句嵌套、多表联合查询、并且支持几乎所有SQL语法的用法，用户问句的表达方式和语义信息也更丰富。但即使作者们考虑到数据集的难度，贴心地将数据集按照难度分为简单、中等和困难，数据集的难度也依然让人望而生畏，目前各项指标也都很低。如何更好地结合数据库信息来理解并表达用户语句的语义、数据库的信息该如何编码及表达、复杂却有必要的SQL语句该如何生成，类似此类的挑战还有很多需要解决，都是非常值得探索的方向。


作者：追一科技
链接：https://www.zhihu.com/question/323109035/answer/683975497 

多表单轮 Spider[13]、
多表多轮 SparC[14]、
对话式 CoSQL[15] 等四个权威的国际公开数据集及榜单

由于表格内容复杂多样，涉及各行各业的专业知识，SQL 的标注难度大且成本高昂，模型迁移能力差，TableQA 一直是自然语言处理领域的难题。

单表单轮：
（单表单轮）存在如下难点：

* 针对简单的 SQL 语句，如何建模 SQL 生成过程； 
* 如何建立自然语言和数据库模式（schema）之间的联系，也称之为模式链接（schema linking）问题。

Text-to-SQL 的目标是将自然语言问题，依据数据库信息（schema) 得到可执行的 SQL。 
多表单轮：
相比单表单轮，多表单轮的任务主要存在以下难点：

* 针对复杂的 SQL 语句，如何设计有语法约束的解码器； 
* 如何利用数据库内多个表格之间的结构信息； 
* 模式链接问题仍然是巨大的挑战，多表情况下更依赖链接的信息进行表格选择。 
对于复杂的 SQL 语句，很难将其拆解为分类问题进行预测，所以学术界又回归到 Seq2Seq 过程，考虑如何设计有约束的生成过程，保证语法正确性。主流的工作在解码的过程中引入 AST (Abstract Syntax Tree) 结构，利用树解码的方式建模语法规则

对于多表之间的结构，主要体现在外键、主键等，这些信息对于模型预测多表联合至关重要。一些研究人员将这种结构抽象为图的形式，并利用图神经网络进行学习

对于模式链接，一些工作开始结合 Transformer 设计更好的模式链接模块，比如 RAT-SQL[18]、LGESQL[19] 等，这些方法更充分地考虑了自然语言中的单词到数据库模式的表、列、值的多粒度链接，极大地提升了模型的性能。近期，人们又开始关注面向表格的预训练模型，模式链接任务也成为重要的预训练目标，利用预训练强大的泛化能力，从而缓解跨领域问题。

大家对于 Spider 的建模主要围绕结构层面展开，之前提到的模式链接其实是自然语言问题和 Schema 之间的结构建模。另外一些工作主要关注于 Schema 内部的建模，将 Schema 中的表、列、外键信息转换为图的形式，融入网络进行学习。而达摩院首先关注到了自然语言问题内部结构对 Text-to-SQL 任务的重要性。

达摩院提出了 SSSQL，将自然语言内部的结构、Schema 内部的结构以及自然语言与 Schema 之间的结构同时建模，并结合一种关系解耦的优化方法，实现了更强的表征能力。

从单轮到多轮

很多情况下，用户需要与表格进行多轮的交互才能完成信息获取，所以 TableQA 进入了第三个阶段，从单轮问答升级为多轮问答，并可以和对话系统进行结合。对于多表多轮，难点主要围绕多轮建模：

* 如何建模多轮用户问题，进行上下文理解； 
* 如何利用历史轮次生成的 SQL，作为重要的信息补充； 
* 如何建模多轮情况下的模式链接问题，涉及到用户话题偏移，非局部依赖等。 
多轮理解一直是对话领域重要的方向，对于多轮问题的建模，通常有直接拼接、轮次 attention 和 gate attention 等方式：


表格预训练模型国内外工作

近年来预训练语言模型（BERT、GPT、T5）迅速发展，促进了 NLP 领域各种任务上的进步，例如阅读理解、命名实体识别等任务。但目前的预训练模型基本上在通用文本上进行训练，在一些需要对结构化表格数据进行建模的任务上（如 Text-to-SQL 和 Table-to-Text），需要同时对结构化数据进行表示，如直接采用现有 BERT 等模型，就面临着编码文本与预训练文本形式不一致的问题。

目前，英文场景已有一些针对结构化数据做预训练的探索（GAP，Grappa），但在中文场景该方向还处于空白状态。

基于此，研究者所在的达摩院 Conversational AI 团队发布了中文首个表格预训练模型 SDCUP，同时也是业界最大表格预训练模型（72 层 Transformer，10 亿参数），在 WikiSQL、SQuALL 等多个学界 Benchmark 均取得 SOTA 效果。


* 单轮模型旨在提升下游的 Text-to-SQL 语义解析任务，代表工作有耶鲁的 Grappa 和亚马逊的 GAP； 
* 多轮模型旨在提升基于表格的对话式语义解析任务（CoSQL），代表工作有微软的 SCORE 和 Element AI 的 PICARD； 
* 生成模型旨在提升 Table-to-Text 和 TableQA 的 Response Generation 生成的效果，代表工作有 Intel 的 TableNLG 和 HIT 的 TableGPT。 
* 

—————
达摩院构建的表格问答中文数据集 TaBLUE 上，SDCUP 比同参数规模 BERT 模型效果提升约 3 个百分点
SDCUP 模型是达摩院表格对话技术系列研发的一部分

团队进一步构建了表格问答中文 Benchmark 数据集 TaBLUE，在基于模板构建的数据基础之上，由人工改写对应的文本，使其更加符合真实的表格问答场景


建模方法

Schema Dependency 系列方法

预训练模型
————————
开源项目

https://github.com/abhijithneilabraham/tableQA


数据集：
https://github.com/yechens/NL2SQL 

NL2SQL：智能对话在打通人与数据查询壁垒上的探索
https://www.modb.pro/db/169906 

https://github.com/ZhuiyiTechnology/TableQA 

https://arxiv.org/abs/2006.06434 

https://github.com/salesforce/WikiSQL 

CHASE : 最新的 Text2SQL 高质量多轮交互数据集


参考文档：
https://www.modb.pro/db/169906 

【综述】
NL2SQL (二) WikiSQL
https://blog.csdn.net/black_soil/article/details/102821562 


背景（why）
现状（公司外，内部）
数据，方法，存在问题（场景），已有产品，
方案：输入，输出，解决那种问题，存在的风险，依赖
FQA：分流方案

NL2SQL实现简述
对于nl2sql的各个系统，在内部实现上，整体结构都大同小异，只是技术不同罢了。简单来说，整个系统将nl2sql分成了SQL几个子句的识别，包括SELECT clause、WHERE clause,当然可能还有group by、limit等等。每个部分又会牵扯很多的细节，比如table识别，属性识别，适当的添加索引等等。图二是采用深度学习方法，通过encoder-decoder的方式进行nl2sql的实现。Google的Analyza采用的则是语义解析和规则的方式构建的，paper中解释主要还是因为数据的问题。


—————

当前技术方案： http://way.xiaojukeji.com/article/30516 

通过基于模板的规则引擎进行解析处理，同时该规则引擎支持横向扩展，可以快速兼容更多样的边界场景；
据检索结果构建Rank模型

TODO：
Text-to-SQL，TableQA 方案调研
配置 luban GPU环境，跑SDCUP预训练模型
智能问答机器人方案设计

查看论文
查看数据集
看代码


预训练表格模型可以分为三大类：单轮、多轮和生成。

* 单轮模型旨在提升下游的 Text-to-SQL 语义解析任务，代表工作有耶鲁的 Grappa 和亚马逊的 GAP； 
* 多轮模型旨在提升基于表格的对话式语义解析任务（CoSQL），代表工作有微软的 SCORE 和 Element AI 的 PICARD； 
* 生成模型旨在提升 Table-to-Text 和 TableQA 的 Response Generation 生成的效果，代表工作有 Intel 的 TableNLG 和 HIT 的 TableGPT。 


预训练表格模型 SDCUP 和相关 NL2SQL 技术已经应用在了阿里云智能客服（云小蜜）的 TableQA 产品中。

表格管理、数据配置、模型训练和效果干预等功能已全部完成产品化，基本做到知识梳理低成本、问答构建高速度、模型训练无标注，满足各个场景的交付运维需求。
Schema Dependency 

预训练表格模型最终的目标是为了提升下游 Text-to-SQL 任务的效果，在自然语言和表格的 schema 之间，存在这一个复杂的语义交互结构（Schema Linking），对于该结构的识别和建模已经成为 Semantic Parsing 任务中的重要瓶颈。然而，业界已有的表格预训练模型没有显式建模自然语言问题和表格数据之间的语义交互结构。

提出了基于模式依存的表格预训练模型，为了提升模型对于不同表格模式下的鲁棒性，还进一步提出了基于模式知识扰动的表格预训练模型；此外，为了减轻数据噪音对模型的影响，团队还提出了基于课程学习的表格预训练模型。


基于模式依存的表格预训练
对于预训练表格模型来说，最关键的问题在于找到自然语言问题和模式之间的关联，又称模式链接问题
基于模式知识的问题扰动任务
基于课程学习的表格预训练
从大规模数据中采样得出具体数据时，如果当前数据的难度小于模型的接受能力，则改数据用来进行训练，反之则放回训练集。随着不断的迭代，所有的数据将渐进式地完成输入。
考虑将“当前数据的难度小于模型的接受能力”改为当前数据的难度与模型的接受能力相当


intent detection
slot filing
https://zhuanlan.zhihu.com/p/261010283 
slot filling and intent detection
SLU（Spoken Language Understanding）一般可以分为两个子任务：槽填充和意图检测。槽填充一般可以看做一个序列标注任务，常用的模型有CRF、RNN等；意图检测可以看做一个分类任务，常用的模型包括SVM、RNN等。两个子任务以前一般分开来做，近年来，提出了很多联合槽填充和意图检测的模型来提高句子级别语言信息的挖掘能力。本文在"基于注意力机制的循环神经网络模型"的基础上，提出了Slot-Gated Mechanism来直接建模意图和槽之间的关系，并取得了比attention-based model更好的表现。

https://blog.csdn.net/fkyyly/article/details/83791833 

1.任务型
对象 任务型导向的对话指特定条件下提供信息或服务的对话。通常情况下是为了满足带有明确目的的用户。
具体场景与功能
例如查流量，查话费，订餐，订票，咨询等任务型场景。由 于用户的需求较为复杂，通常情况下需分多轮陈述，用户也可能在对话过程中不断修改与完善自己的需求，任务型机器人需要通过询问、澄清和确认来帮助用户明确目的。
任务评价
首先要有解决问题的明确目标，评价指标较重要的一点为，轮次尽可能少，尽量直逼答案 ，如果答非所问严重影响用户使用。


任务型导向的对话目的十分明确，关键是获取意图与约束条件（槽位）以及对话状态的追踪。

任务型导向对话在大家族中的位置
￼
这里的分类是这样分的：首先将对话分为问答与会话，在问答中按照文档是否结构化分为无结构化文档与结构化文档。无结构化文档中包含一些如IR信息检索（如QA对，查找文档的问题），IE信息抽取（如阅读理解，查找文档中的精确片段），这一块的难点在于相似性的计算。结构化文档中包含数据库，知识图谱等，他们的输入为结构化的片段，数据库具有查询的功能，知识图谱具有查询与推理的能力，这一块的难点其实也是如何获取自然语言中的约束条件（槽位）的问题。接下来重点看下会话，会话划为为闲聊型，任务型等，传统的任务型分为语言理解模块（SLU），对话管理模块（DM）以及自然语言生成模块（NLG）等。后续的介绍主要针对SLU模块中的联合模型的一个介绍。

2. 任务型中的语义表示
自然语言如何解析成合适的语义表示一直是一个难题。下面主要介绍三种相关的语义表示方法。
1. 分布语义表示(Distributional semantics) 主要包括词level与句子level。词level上主要有：Word2Vector, GloVe, ELMo, FastText...等句子level主要有Skip-Thoughts, Quick-Thoughts, InferSent...等分布式表示
2. 框架语义表示(Frame semantics) 主要包括三元组的形式，Action(slot,value)，例如查询币种，Inform(货币=人民币,...)
3. 模型论语义表示(Model-theoretic semantics) 这是十分有趣的一种表示方式，参见Paul Gochet著作《Philosophical Perspectives for Pragmatics》。

我们这里以一个例子开始语言理解模块的拓展。对于一段对话我们需要通过语言理解模块对它进行解析，包括领域的识别如是航空还是酒店，以及每个片段的意图，比如是购票还是退票，以及每个具体意图下的约束信息（槽位）。

3. 语言理解模块
语言理解模块主要包括意图与槽位的识别。意图的识别实际上是一个分类问题，如基于规则，传统机器学习算法 (SVM)，基于深度学习算法（CNN, LSTM, RCNN, C-LSTM, FastText）等。意图在对话中还涉及转换，这里不在说明。Slot识别实际上是一种序列标记的任务，如基于规则 (Phoenix Parser)，基于传统机器学习算法 (DBN; SVM)，基于深度学习算法（LSTM, Bi-RNN, Bi-LSTM-CRF）。有些人可能不大区别slot与实体的区别，下面以BIO标注举个例子：
如”show flights from Boston to New York today“中对于实体来讲Boston与New York都是标注为city，而对于slot来说区分为出发城市与目的地城市，可以这样说槽位的种类相比与实体更加的多元化。

意图与槽位的联合模型

Joint Model (Intent+Slot)
1. 第一篇文章主要利用双向的GRU+CRF作为意图与槽位的联合模型。

autoNLP
https://www.kdnuggets.com/2021/06/overview-autonlp-hugging-face-example-project.html 


way文档
http://way.xiaojukeji.com/article/30516 