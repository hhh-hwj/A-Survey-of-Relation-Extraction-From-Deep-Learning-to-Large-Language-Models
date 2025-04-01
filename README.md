
# A Survey of Relation Extraction: From Deep Learning to Large Language Models

<p align="center" width="80%">
<img src=".文章结构图.PNG" style="width: 50%">
</p>

The organization of papers is discussed in our survey: A Survey of Relation Extraction: From Deep Learning to Large Language Models]


    
## 📒 Table of Contents
- [Datasets](#Datasets)
- [Relation extraction based on deep learning](#Relation extraction based on deep learning)
    - [Pipeline-based Methods](#Pipeline-based Methods)
    - [Joint Extraction-based Methods](#Joint Extraction-based Methods)
- [Relation extraction based on LLMs](#specific-domain)
     - [Background of LLMs](#supervised-fine-tuning)
     - [Prompt-based Methods](#supervised-fine-tuning)
     - [Fine-tuning-based Methods](#supervised-fine-tuning)
     - [Data Augmentation-based Methods](#supervised-fine-tuning)
- [Relation Extraction Open-source Tools](#evaluation-and-analysis)


# Datasets
\* denotes the dataset is multimodal. # refers to the number of categories or sentences.

<table>
    <thead>
        <tr>
            <th align="center">Dataset</th>
            <th align="center">Domain</th>
            <th align="center">#Class</th>
            <th align="center">#Train</th>
            <th align="center">#Val</th>
            <th align="center">#Test</th>
            <th align="center">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">SemEval-2010 Task 8</td>
            <td align="center">General</td>
            <td align="center">19</td>
            <td align="center">6507</td>
            <td align="center">-</td>
            <td align="center">2717</td>
            <td align="center"><a href="https://aclanthology.org/S10-1006/">Link</a></td>
        </tr>
        <tr>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">6</td>
            <td align="center">10051</td>
            <td align="center">2424</td>
            <td align="center">2050</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL04</td>
            <td align="center">News</td>
            <td align="center">6</td>
            <td align="center">1153</td>
            <td align="center">288</td>
            <td align="center">288</td>
            <td align="center"><a href="https://cogcomp.seas.upenn.edu/page/resource view/43">Link</a></td>
        </tr>
        <tr>
            <td align="center">NYT</td>
            <td align="center">News</td>
            <td align="center">24</td>
            <td align="center">56195</td>
            <td align="center">5000</td>
            <td align="center">5000</td>
            <td align="center"><a href="https://iesl.cs.umass.edu/riedel/ecml/">Link</a></td>
        </tr>
        <tr>
            <td align="center">WebNLG</td>
            <td align="center">General</td>
            <td align="center">246</td>
            <td align="center">5019</td>
            <td align="center">500</td>
            <td align="center">703</td>
            <td align="center"><a href="https://github.com/weizhepei/CasRel/tree/master/data/WebNLG">Link</a></td>
        </tr>
        <tr>
            <td align="center">ADE</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">3417</td>
            <td align="center">427</td>
            <td align="center">428</td>
            <td align="center"><a href="https://sites.google.com/site/adecorpus/">Link</a></td>
        </tr>
        <tr>
            <td align="center">SciERC</td>
            <td align="center">Scientific</td>
            <td align="center">7</td>
            <td align="center">1861</td>
            <td align="center">275</td>
            <td align="center">551</td>
            <td align="center"><a href="https://nlp.cs.washington.edu/sciIE/">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACRED</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://nlp.stanford.edu/projects/tacred/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Re-TACRED</td>
            <td align="center">News</td>
            <td align="center">40</td>
            <td align="center">58465</td>
            <td align="center">19584</td>
            <td align="center">13418</td>
            <td align="center"><a href="https://github.com/gstoica27/Re-TACRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACREV</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://github.com/DFKI-NLP/tacrev">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACREV</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://github.com/DFKI-NLP/tacrev">Link</a></td>
        </tr>
    </tbody>
</table>


# Relation extraction based on deep learning 
This section provides a detailed introduction to two deep learning-based RE methods: pipeline-based and joint extraction-based methods.
## Pipeline-based Methods
|  Paper  |      Dataset     | Code |
| :----- | :--------------: | :---------: |
|  [Relation classification via convolutional deep neural network](https://aclanthology.org/C14-1220/)  |   SemEval-2010 Task 8      | [GitHub](https://github.com/onehaitao/CNN-relation-extraction)|
|  [Relation extraction: Perspective from convolutional neural networks](https://aclanthology.org/W15-1506/)  | SemEval-2010 Task 8  |
|  [Classifying relations by ranking with convolutional neural networks](https://aclanthology.org/P15-1061/)  | SemEval-2010 Task 8   |[GitHub](https://github.com/onehaitao/CR-CNN-relation-extraction)|
|  [Relation classification via multi-level attention CNNs](https://aclanthology.org/P16-1123/)  |  SemEval-2010 Task 8   |[GitHub](https://github.com/FrankWork/acnn)|
|  [Semantic relation classification via convolutional neural networks with simple negative sampling](https://aclanthology.org/D15-1062/) |   SemEval-2010 Task 8    | 
|  [Knowledge-oriented convolutional neural network for causal relation extraction from natural language texts](https://www.sciencedirect.com/science/article/abs/pii/S0957417418305177)  |   SemEval-2010 Task 8   |
|  [Semantic compositionality through recursive matrix-vector spaces](https://aclanthology.org/D12-1110/)|   SemEval-2010 Task 8    |    
|  [Simple customization of recursive neural networks for semantic relation classification](https://aclanthology.org/D13-1137/) |   SemEval-2010 Task 8    |
|  [Chain based rnn for relation classification](https://aclanthology.org/N15-1133.pdf) |   SemEval-2010 Task 8    |
|  [Improved relation classification by deep recurrent neural networks with data augmentation](https://aclanthology.org/C16-1138/) |   SemEval-2010 Task 8    |      
|  [Classifying relations via long short term memory networks along shortest dependency paths](https://aclanthology.org/D15-1206/) |   SemEval-2010 Task 8    |
|  [Attention-based bidirectional long short-term memory networks for relation classification](https://aclanthology.org/P16-2034/))|   SemEval-2010 Task 8     |
|  [Attention-based lstm with filter mechanism for entity relation classification](https://www.mdpi.com/2073-8994/12/10/1729)|   SemEval-2010 Task 8     |
|  [Direction-sensitive relation extraction using bi-sdp attention model](https://www.sciencedirect.com/science/article/abs/pii/S0950705120302628) |SemEval-2010 Task 8 |
|  [A dependency-based neural network for relation classification](https://aclanthology.org/P15-2047.pdf) |   SemEval-2010 Task 8    |
|  [Bidirectional recurrent convolutional neural network for relation classification](https://aclanthology.org/P16-1072/) |   SemEval-2010 Task 8    |
|  [Structure regularized neural network for entity relation classification for Chinese literature text](https://aclanthology.org/N18-2059/) |  self-control    |
|  [Neural relation classification with text descriptions](https://aclanthology.org/C18-1100/) |   SemEval-2010 Task 8    |
|  [A combination of rnn and cnn for attention-based relation classification](https://www.sciencedirect.com/science/article/pii/S187705091830601X) | SemEval-2010 Task 8 |
|  [A single attention-based combination of cnn and rnn for relation classification](https://ieeexplore.ieee.org/document/8606107) |   SemEval-2010 Task 8    |
|  [Relation classification using segment-level attention-based CNN and dependency-based RNN](https://aclanthology.org/N19-1286/) |   SemEval-2010 Task 8     |
## Joint Extraction-based Methods
|  Paper  |      Dataset     | Code |
| :----- | :--------------: | :---------: |
|  [End-to-end relation extraction using LSTMs on sequences and tree structures](https://aclanthology.org/P16-1105/) |   ACE05、ACE04     | [GitHub](https://github.com/tticoin/LSTM-ER)|
|  [Going out on a limb: Joint extraction of entity mentions and relations without dependency trees](https://aclanthology.org/P17-1085/)  |  ACE05  |
|  [Joint entity and relation extraction based on a hybrid neural network](https://www.sciencedirect.com/science/article/abs/pii/S0925231217301613) | ACE05  |
|  [Graphrel: Modeling text as relational graphs for joint entity and relation extraction](https://aclanthology.org/P19-1136/) |  NYT、WebNLG   |[GitHub](https://github.com/tsujuifu/pytorch_graph-rel)|
|  [A relational adaptive neural model for joint entity and relation extraction](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.635492/full) |    NYT、WebNLG     | 
|  [Entity-relation extraction as multi-turn question answering](https://aclanthology.org/P19-1129/)  |  ACE04、ACE05、CoNLL04   |[GitHub](https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA)|
|  [Two are better than one: Joint entity and relation extraction with table-sequence encoders](https://aclanthology.org/2020.emnlp-main.133/)| ACE04、ACE05、CoNLL04、ADE  |   [GitHub](https://github.com/LorrinWWW/two-are-better-than-one)|
|  [Joint extraction of entities and overlapping relations using source-target entity labeling](https://www.sciencedirect.com/science/article/abs/pii/S0957417421002943) |   NYT、WebNLG    |
|  [A multigate encoder for joint entity and relation extraction](https://aclanthology.org/2022.ccl-1.75/) |   ACE04、ACE05、SciERC   |
|  [Prta:joint extraction of medical nested entities and overlapping relation via parameter sharing progressive recognition and targeted assignment decoding scheme](https://www.sciencedirect.com/science/article/abs/pii/S0010482524006231) |   NYT、ACE04、 ACE05   |      
|  [Joint extraction of entities and relations based on a novel tagging scheme](https://aclanthology.org/P17-1113/) |    NYT   |[GitHub](https://github.com/gswycf/Joint-Extraction-of-Entities-and-Relations-Based-on-a-Novel-Tagging-Scheme)|
|  [Joint extraction of entities and overlapping relations using position-attentive sequence labeling](https://ojs.aaai.org/index.php/AAAI/article/view/4591))|   NYT  |
|  [Joint extraction of entities and relations based on a novel decomposition strategy](https://www.nature.com/articles/s41598-024-51559-w)|    NYT、WebNLG     |[GitHub](https://github.com/yubowen-ph/JointER)|
|  [A novel cascade binary tagging framework for relational triple extraction](https://aclanthology.org/2020.acl-main.136/) |NYT、WebNLG |[GitHub](https://github.com/weizhepei/CasRel)|
|  [Tplinker: Single-stage joint extraction of entities and relations through token pair linking](https://aclanthology.org/2020.coling-main.138/) |   NYT、WebNLG |[GitHub](https://github.com/131250208/TPlinker-joint-extraction)|
|  [PRGC: Potential relation and global correspondence based joint relational triple extraction](https://aclanthology.org/2021.acl-long.486/) |  NYT、WebNLG     |[GitHub](https://github.com/hy-struggle/PRGC)|
|  [A simple overlapping relation extraction method based on dropout](https://ieeexplore.ieee.org/document/9892310) |   NYT、WebNLG    |
|  [Onerel: Joint entity and relation extraction with one module in one step](https://arxiv.org/abs/2203.05412) | NYT、WebNLG  |[GitHub](https://github.com/China-ChallengeHub/OneRel)|
## Relation extraction based on LLMs 
Models targeting only RE tasks.
### Background of LLMs
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
| [Enhancing Software-Related Information Extraction via Single-Choice Question Answering with Large Language Models](https://arxiv.org/pdf/2404.05587)  | Others | 2024-04 | []() |
| [CRE-LLM: A Domain-Specific Chinese Relation Extraction Framework with Fine-tuned Large Language Model](https://arxiv.org/abs/2404.18085)  | Arxiv | 2024-04 | [GitHub](https://github.com/SkyuForever/CRE-LLM) |
| [Recall, Retrieve and Reason: Towards Better In-Context Relation Extraction](https://arxiv.org/abs/2404.17809)  | IJCAI | 2024-04 | []() |
| [Empirical Analysis of Dialogue Relation Extraction with Large Language Models](https://arxiv.org/abs/2404.17802)  | IJCAI | 2024-04 | []() |
| [Meta In-Context Learning Makes Large Language Models Better Zero and Few-Shot Relation Extractors](https://arxiv.org/abs/2404.17807)  | IJCAI | 2024-04 | []() |
| [Retrieval-Augmented Generation-based Relation Extraction](https://arxiv.org/abs/2404.13397)  | Arxiv | 2024-04 | [GitHub](https://github.com/sefeoglu/RAG4RE) |
| [Relation Extraction Using Large Language Models: A Case Study on Acupuncture Point Locations](https://arxiv.org/pdf/2404.05415)  | Arxiv | 2024-04 | []() |
|  [STAR: Boosting Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29839)  |  AAAI  |  2024-03 |  |
| [Grasping the Essentials: Tailoring Large Language Models for Zero-Shot Relation Extraction](https://arxiv.org/abs/2402.11142) | Arxiv | 2024-02 |   
| [Chain of Thought with Explicit Evidence Reasoning for Few-shot Relation Extraction](https://aclanthology.org/2023.findings-emnlp.153/) | EMNLP Findings | 2023-12 | |
|  [GPT-RE: In-context Learning for Relation Extraction using Large Language Models](https://arxiv.org/abs/2305.02105)  |   EMNLP  |  2023-12   | [GitHub](https://github.com/YukinoWan/GPT-RE) |
|  [Guideline Learning for In-context Information Extraction](https://arxiv.org/abs/2310.05066)  |   EMNLP  |  2023-12   |  |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |   EMNLP Findings | 2023-12 | [GitHub](https://github.com/mayubo2333/LLM-IE) |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings    |  2023-12   | [GitHub](https://github.com/ridiculouz/LLMAAA) |
|  [Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs](https://arxiv.org/abs/2312.00552)  |   EMNLP    |  2023-12   | [GitHub](https://github.com/qingwang-isu/AugURE) |
|  [Revisiting Large Language Models as Zero-shot Relation Extractors](https://arxiv.org/abs/2310.05028)  |   EMNLP Findings |  2023-12   |  |
| [Mastering the Task of Open Information Extraction with Large Language Models and Consistent Reasoning Environment](https://arxiv.org/abs/2310.10590) | Arxiv | 2023-10 | 
|  [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://aclanthology.org/2023.findings-acl.50.pdf)  |  ACL Findings  |  2023-07   | [GitHub](https://github.com/OSU-NLP-Group/QA4RE) |
|  [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](https://arxiv.org/abs/2305.01555)  |   ACL Workshop  |  2023-07   | [GitHub](https://github.com/zjunlp/DeepKE/tree/main/example/llm/UnleashLLMRE) |
| [Sequence generation with label augmentation for relation extraction](https://ojs.aaai.org/index.php/AAAI/article/view/26532)| AAAI | 2023-06 | [GitHub](https://github.com/pkuserc/RELA) |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Arxiv    |  2023-04   |  |
| [DORE: Document Ordered Relation Extraction based on Generative Framework](https://aclanthology.org/2022.findings-emnlp.253/) |  EMNLP Findings | 2022-12 |
|  [REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204/)  |    EMNLP Findings      |   2021-11   | [GitHub](https://github.com/babelscape/rebel) |
### Prompt-based Methods
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
| [ERA-CoT: Improving Chain-of-Thought through Entity Relationship Analysis](https://aclanthology.org/2024.acl-long.476/) |  ACL | 2024 | [GitHub](https://github.com/OceannTwT/era-cot) |
| [AutoRE: Document-Level Relation Extraction with Large Language Models](https://aclanthology.org/2024.acl-demos.20/) | ACL Demos | 2024 | [GitHub](https://github.com/bigdante/AutoRE) |
| [Meta In-Context Learning Makes Large Language Models Better Zero and Few-Shot Relation Extractors](https://arxiv.org/abs/2404.17807)  | IJCAI | 2024-04 | []() |
| [Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction](https://arxiv.org/abs/2401.13598) | WWW | 2024 |
| [Improving Recall of Large Language Models: A Model Collaboration Approach for Relational Triple Extraction](https://aclanthology.org/2024.lrec-main.778/)  | COLING | 2024 | [GitHub](https://github.com/Ding-Papa/Evaluating-filtering-coling24) |
| [Unlocking Instructive In-Context Learning with Tabular Prompting for Relational Triple Extraction](https://aclanthology.org/2024.lrec-main.1488/) | COLING | 2024 | |
| [A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction](https://arxiv.org/abs/2402.13364) |  Arxiv | 2024-02 | 
| [Structured information extraction from scientific text with large language models](https://www.nature.com/articles/s41467-024-45563-x) | Nature Communications | 2024-02 | [GitHub](https://github.com/lbnlp/nerre-llama) |
| [Document-Level In-Context Few-Shot Relation Extraction via Pre-Trained Language Models](https://arxiv.org/abs/2310.11085) | Arxiv | 2024-02 | [GitHub](https://github.com/oezyurty/REPLM) |
| [Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction](https://arxiv.org/abs/2402.14373) | Arxiv | 2024-02 | |
| [Efficient Data Learning for Open Information Extraction with Pre-trained Language Models](https://aclanthology.org/2023.findings-emnlp.869/) | EMNLP Findings | 2023-12 | 
| [Mastering the Task of Open Information Extraction with Large Language Models and Consistent Reasoning Environment](https://arxiv.org/abs/2310.10590) | Arxiv | 2023-10 | 
| [Unified Text Structuralization with Instruction-tuned Language Models](https://arxiv.org/abs/2303.14956)  | Arxiv | 2023-03 | []() |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |    EMNLP      | 2021-11    | [GitHub](https://github.com/PlusLabNLP/TempGen) |

### Fine-tuning-based Methods
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
| [MetaIE: Distilling a Meta Model from LLM for All Kinds of Information Extraction Tasks](https://arxiv.org/abs/2404.00457)  | Arxiv | 2024-03 | [GitHub](https://github.com/KomeijiForce/MetaIE) |
| [Distilling Named Entity Recognition Models for Endangered Species from Large Language Models](https://arxiv.org/abs/2403.15430)  | Arxiv | 2024-03 | []() |
| [CHisIEC: An Information Extraction Corpus for Ancient Chinese History](https://aclanthology.org/2024.lrec-main.283/)  | COLING | 2024-03 | [GitHub](https://github.com/tangxuemei1995/CHisIEC) |
| [An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction](https://ojs.aaai.org/index.php/AAAI/article/view/29919) | AAAI | 2024-03 | [GitHub](https://github.com/urchade/ATG) |
| [C-ICL: Contrastive In-context Learning for Information Extraction](https://arxiv.org/abs/2402.11254) | Arxiv | 2024-02 | 
|  [REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204/)  |    EMNLP Findings      |   2021-11   | [GitHub](https://github.com/babelscape/rebel) |

## Relation Extraction Open-source Tools 
Models targeting only EE tasks.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quqxui/Awesome-LLM4IE-Papers&type=Date)](https://star-history.com/#quqxui/Awesome-LLM4IE-Papers&Date)
