
# A Survey of Relation Extraction: From Deep Learning to Large Language Models

<p align="center" width="80%">
<img src="Article Structure.png" style="width: 50%">
</p>

The organization of papers is discussed in our survey: A Survey of Relation Extraction: From Deep Learning to Large Language Models]


    
## 📒 Table of Contents
- [Problem Definition](#Problem-Definition)
- [Datasets](#Datasets)
- [Evaluation Metrics](#Evaluation-Metrics)
- [Relation extraction based on deep learning](#Relation-extraction-based-on-deep-learning)
    - [Pipeline-based Methods](#Pipeline-based-Methods)
    - [Joint Extraction-based Methods](#Joint-Extraction-based-Methods)
- [Relation extraction based on LLMs](#Relation-extraction-based-on-LLMs)
     - [Prompt-based Methods](#Prompt-based-Methods)
     - [Fine-tuning-based Methods](#Fine-tuning-based-Methods)
- [Multi-domain relation extraction](#Multi-domain-relation-extraction)
- [Relation Extraction Open-source Tools](#Relation-Extraction-Open-source-Tools)

# Problem Definition
**Relation Classification** refers to the identification of the relation type between two given entities.  

**Relation Triplet** refers to the identification of the relation type and the corresponding head and tail entity spans.  

**Relation Strict** refers to the identification of the correct relation type, the span of the entities, and the type of head and tail entities.
# Datasets
We have summarized the commonly used datasets for RE tasks through an extensive survey of the existing literature.To assist readers, the table includes the domain, dataset sizes, and corresponding links for each dataset.

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
            <td align="center">1493</td>
            <td align="center">2717</td>
            <td align="center"><a href="https://aclanthology.org/S10-1006/">Link</a></td>
        </tr>
        <tr>
            <td align="center">ACE05</td>
            <td align="center">General</td>
            <td align="center">6</td>
            <td align="center">10051</td>
            <td align="center">2424</td>
            <td align="center">2050</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL04</td>
            <td align="center">General</td>
            <td align="center">6</td>
            <td align="center">1153</td>
            <td align="center">288</td>
            <td align="center">288</td>
            <td align="center"><a href="https://cogcomp.seas.upenn.edu/page/resource view/43">Link</a></td>
        </tr>
        <tr>
            <td align="center">NYT</td>
            <td align="center">General</td>
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
            <td align="center">General</td>
            <td align="center">7</td>
            <td align="center">1861</td>
            <td align="center">275</td>
            <td align="center">551</td>
            <td align="center"><a href="https://nlp.cs.washington.edu/sciIE/">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACRED</td>
            <td align="center">General</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://nlp.stanford.edu/projects/tacred/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Re-TACRED</td>
            <td align="center">General</td>
            <td align="center">40</td>
            <td align="center">58465</td>
            <td align="center">19584</td>
            <td align="center">13418</td>
            <td align="center"><a href="https://github.com/gstoica27/Re-TACRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACREV</td>
            <td align="center">General</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://github.com/DFKI-NLP/tacrev">Link</a></td>
        </tr>
        <tr>
            <td align="center">DocRED</td>
            <td align="center">General</td>
            <td align="center">96</td>
            <td align="center">3053</td>
            <td align="center">1000</td>
            <td align="center">1000</td>
            <td align="center"><a href="https://github.com/thunlp/DocRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">Re-DocRED </td>
            <td align="center">Generall</td>
            <td align="center">96</td>
            <td align="center">3053</td>
            <td align="center">500</td>
            <td align="center">500</td>
            <td align="center"><a href="https://github.com/tonytan48/Re-DocRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">CodRED</td>
            <td align="center">General</td>
            <td align="center">276</td>
            <td align="center">2733</td>
            <td align="center">1010</td>
            <td align="center">1012</td>
            <td align="center"><a href="https://github.com/thunlp/CodREDv">Link</a></td>
        </tr>
        <tr>
            <td align="center">DocRED-HWE</td>
            <td align="center">General</td>
            <td align="center">96</td>
            <td align="center">3053</td>
            <td align="center">699</td>
            <td align="center">1000</td>
            <td align="center"><a href="https://github.com/Hytn/DocRED-HWE">Link</a></td>
        </tr>
        <tr>
            <td align="center">REFinD</td>
            <td align="center">Finance </td>
            <td align="center">22</td>
            <td align="center">20070</td>
            <td align="center">4306</td>
            <td align="center">4300</td>
            <td align="center"><a href="https://refind-re.github.io/index.html">Link</a></td>
        </tr>
        <tr>
            <td align="center">MNRE</td>
            <td align="center">Multi-modal</td>
            <td align="center">23</td>
            <td align="center">12247</td>
            <td align="center">1624</td>
            <td align="center">1614</td>
            <td align="center"><a href="https://github.com/thecharm/MNRE">Link</a></td>
        </tr>
        <tr>
            <td align="center">CAIL</td>
            <td align="center">Legal</td>
            <td align="center">4 </td>
            <td align="center">1400</td>
            <td align="center">-</td>
            <td align="center">250</td>
            <td align="center"><a href="https://github.com/DUTIR-LegalIntelligence/JointExtraction4Legal">Link</a></td>
        </tr>
        <tr>
            <td align="center">FewRel</td>
            <td align="center">General</td>
            <td align="center">80</td>
            <td align="center"> 52500</td>
            <td align="center">-</td>
            <td align="center">3500</td>
            <td align="center"><a href="https://github.com/thunlp/FewRel">Link</a></td>
        </tr>
        <tr>
            <td align="center">Wiki-ZSL</td>
            <td align="center">General</td>
            <td align="center">113</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center"><a href="https://github.com/dinobby/ZS-BERT">Link</a></td>
        </tr>
        <tr>
            <td align="center">ADE</td>
            <td align="center"> Biomedical</td>
            <td align="center">1 </td>
            <td align="center">3417 </td>
            <td align="center"> 427</td>
            <td align="center">428</td>
            <td align="center"><a href="https://sites.google.com/site/adecorpus/">Link</a></td>
        </tr>
        <tr>
            <td align="center">DDI2013 </td>
            <td align="center">Biochemical </td>
            <td align="center">4</td>
            <td align="center">6976</td>
            <td align="center">-</td>
            <td align="center">1299</td>
            <td align="center"><a href="https://github.com/albertrial/SemEval-2013-task-9">Link</a></td>
        </tr>
    </tbody>
</table>

# Evaluation Metrics
The core metrics for evaluating the performance of RE systems include precision (P), recall (R), and F1-score, which encompasses both Macro-F1 and Micro-F1. Precision is the proportion of true positives among all samples predicted as positive, while recall is the proportion of true positives correctly identified among all actual positives. The F1 score is the harmonic mean of precision and recall, balancing the trade-off between them. Micro-F1 is better for imbalanced datasets, as it aggregates true positive (TP), false positive (FP), and false negative (FN) across all classes, while Macro-F1 is more suited for balanced datasets, giving equal weight to each class. TP refers to samples correctly predicted as positive, FP refers to those incorrectly predicted as positive, and FN refers to actual positive samples predicted as negative, with \( \text{F1}_i \) representing the F1 score of the \( i \)-th class. The formulas for calculating the aforementioned metrics are given in equations (1) to (7).

### Precision (P)
$$
\text{Precision (P)} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

### Recall (R)
$$
\text{Recall (R)} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

### F1 Score
$$
\text{F1} = 2 \times \frac{\text{P} \times \text{R}}{\text{P} + \text{R}}
$$

### Macro-F1
$$
\text{Macro-F1} = \frac{1}{N} \sum_{i=1}^{N} \text{F1}_i
$$

### Micro-Precision (Micro-P)
$$
\text{Micro-Precision (Micro-P)} = \frac{\sum_{i} \text{TP}_i}{\sum_{i} \text{TP}_i + \sum_{i} \text{FP}_i}
$$

### Micro-Recall (Micro-R)
$$
\text{Micro-Recall (Micro-R)} = \frac{\sum_{i} \text{TP}_i}{\sum_{i} \text{TP}_i + \sum_{i} \text{FN}_i}
$$

### Micro-F1
$$
\text{Micro-F1} = \frac{2 \times \text{Micro-P} \times \text{Micro-R}}{\text{Micro-P} + \text{Micro-R}}
$$

# Pre-trained Model
We have summarized the commonly used datasets for RE tasks through an extensive survey of the existing literature.To assist readers, the table includes the domain, dataset sizes, and corresponding links for each dataset.

<table>
    <thead>
        <tr>
            <th align="center">Model</th>
            <th align="center">Size</th>
            <th align="center">Base model</th>
            <th align="center">Release Time</th>
            <th align="center">Open Source</th>
            <th align="center">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">BERT-base</td>
            <td align="center">110M</td>
            <td align="center">-</td>
            <td align="center">2018</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/google-bert/bert-base-uncased">Link</a></td>
        </tr>
        <tr>
            <td align="center">RoBERTa-large</td>
            <td align="center">355M</td>
            <td align="center">-</td>
            <td align="center">2019</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/FacebookAI/roberta-large">Link</a></td>
        </tr>
         <tr>
            <td align="center">BART-base</td>
            <td align="center">139M</td>
            <td align="center">-</td>
            <td align="center">2019</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/facebook/bart-base">Link</a></td>
        </tr>
        <tr>
            <td align="center">T5-large</td>
            <td align="center">738M</td>
            <td align="center">-</td>
            <td align="center">2020</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/google-t5/t5-large">Link</a></td>
        </tr>
        <tr>
            <td align="center">Flan T5-XL </td>
            <td align="center">2.85B</td>
            <td align="center">T5</td>
            <td align="center">2022</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/google/flan-t5-xl">Link</a></td>
        </tr>
        <tr>
            <td align="center">CodeLlama-34B</td>
            <td align="center">33.7B</td>
            <td align="center">llama2</td>
            <td align="center">2023</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/meta-llama/CodeLlama-34b-hf">Link</a></td>
        </tr>
        <tr>
            <td align="center">Mistral-7B</td>
            <td align="center">7.24B</td>
            <td align="center">-</td>
            <td align="center">2023</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/mistralai">Link</a></td>
        </tr>
        <tr>
            <td align="center">LLaMA3.1-8B</td>
            <td align="center">8.03B</td>
            <td align="center">-</td>
            <td align="center">2024</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">Link</a></td>
        </tr>
         <tr>
            <td align="center">GPT-3</td>
            <td align="center">175B</td>
            <td align="center">-</td>
            <td align="center">2020</td>
            <td align="center">×</td>
            <td align="center"><a href="https://openai.com/api/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Text-davinci-003</td>
            <td align="center">-</td>
            <td align="center">GPT-3</td>
            <td align="center">2022</td>
            <td align="center">×</td>
            <td align="center"><a href="https://openai.com/api/">Link</a></td>
        </tr>
        <tr>
            <td align="center">GPT-3.5-turbo series</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">2020</td>
            <td align="center">×</td>
            <td align="center"><a href="https://openai.com/api/">Link</a></td>
        </tr>
        <tr>
            <td align="center">GPT-4 series</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">2023</td>
            <td align="center">×</td>
            <td align="center"><a href="https://openai.com/api/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Deepseek-R1 series</td>
            <td align="center">685B</td>
            <td align="center">-</td>
            <td align="center">2025</td>
            <td align="center">✓</td>
            <td align="center"><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">Link</a></td>
        </tr>
        <tr>
            <td align="center">Gemini series</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">2023</td>
            <td align="center">×</td>
            <td align="center"><a href="https://deepmind.google/technologies/gemini/">Link</a></td>
        </tr>
    </tbody>
</table>

# Relation extraction based on deep learning 
This section provides a detailed introduction to two deep learning-based RE methods: pipeline-based and joint extraction-based methods.
## Pipeline-based Methods
|  Paper  |      Dataset | Evaluation Metrics    | Code |
| :----- | :--------------: |:---------: | :---------: |
|  [Relation classification via convolutional deep neural network](https://aclanthology.org/C14-1220/)  |   SemEval     | Macro-F1     |  [GitHub](https://github.com/onehaitao/CNN-relation-extraction)|
|  [Relation extraction: Perspective from convolutional neural networks](https://aclanthology.org/W15-1506/)  | SemEval  |P,R,Macro-F1     | 
|  [Classifying relations by ranking with convolutional neural networks](https://aclanthology.org/P15-1061/)  | SemEval   | P,R,Macro-F1      | [GitHub](https://github.com/onehaitao/CR-CNN-relation-extraction)|
|  [Relation classification via multi-level attention CNNs](https://aclanthology.org/P16-1123/)  |  SemEval   |  Macro-F1     | [GitHub](https://github.com/FrankWork/acnn)|
|  [Semantic relation classification via convolutional neural networks with simple negative sampling](https://aclanthology.org/D15-1062/) |   SemEval    | Macro-F1       |  
|  [Knowledge-oriented convolutional neural network for causal relation extraction from natural language texts](https://www.sciencedirect.com/science/article/abs/pii/S0957417418305177)  |  SemEval     |   P,R,Macro-F1    | 
|  [Semantic compositionality through recursive matrix-vector spaces](https://aclanthology.org/D12-1110/)|   SemEval   |    Macro-F1     |   
|  [Simple customization of recursive neural networks for semantic relation classification](https://aclanthology.org/D13-1137/) |   SemEval   |Macro-F1      | 
|  [Chain based rnn for relation classification](https://aclanthology.org/N15-1133.pdf) |   SemEval    | Macro-F1    | 
|  [Improved relation classification by deep recurrent neural networks with data augmentation](https://aclanthology.org/C16-1138/) |   SemEval  | Macro-F1     | 
|  [Classifying relations via long short term memory networks along shortest dependency paths](https://aclanthology.org/D15-1206/) |   SemEval    |Macro-F1    | 
|  [Attention-based bidirectional long short-term memory networks for relation classification](https://aclanthology.org/P16-2034/))|   SemEval  | Macro-F1     | 
|  [Attention-based lstm with filter mechanism for entity relation classification](https://www.mdpi.com/2073-8994/12/10/1729)|   SemEval    | Macro-F1    | 
|  [Direction-sensitive relation extraction using bi-sdp attention model](https://www.sciencedirect.com/science/article/abs/pii/S0950705120302628) |SemEval | Macro-F1     | 
|  [A dependency-based neural network for relation classification](https://aclanthology.org/P15-2047.pdf) |   SemEval    | Macro-F1     | 
|  [Bidirectional recurrent convolutional neural network for relation classification](https://aclanthology.org/P16-1072/) |   SemEval  | Macro-F1      | 
|  [Neural relation classification with text descriptions](https://aclanthology.org/C18-1100/) |   SemEval   | P,R,Macro-F1     | 
|  [A combination of rnn and cnn for attention-based relation classification](https://www.sciencedirect.com/science/article/pii/S187705091830601X) | SemEval | Macro-F1       | 
|  [A single attention-based combination of cnn and rnn for relation classification](https://ieeexplore.ieee.org/document/8606107) |   SemEval |  Macro-F1    | 
|  [Relation classification using segment-level attention-based CNN and dependency-based RNN](https://aclanthology.org/N19-1286/) |   SemEval  |  Macro-F1     | 
## Joint Extraction-based Methods
|  Paper  |      Dataset     |Evaluation Metrics    | Code |
| :----- | :--------------: | :---------: |:---------: |
|  [End-to-end relation extraction using LSTMs on sequences and tree structures](https://aclanthology.org/P16-1105/) |   ACE05、ACE04、SemEval    |P,R,Micro-F1,Macro-F1 | [GitHub](https://github.com/tticoin/LSTM-ER)|
|  [Going out on a limb: Joint extraction of entity mentions and relations without dependency trees](https://aclanthology.org/P17-1085/)  |  ACE05  | P,R,Micro-F1  |
|  [Joint entity and relation extraction based on a hybrid neural network](https://www.sciencedirect.com/science/article/abs/pii/S0925231217301613) | ACE05  | P,R,Micro-F1  |
|  [Graphrel: Modeling text as relational graphs for joint entity and relation extraction](https://aclanthology.org/P19-1136/) |  NYT、WebNLG   | P,R,Micro-F1  |[GitHub](https://github.com/tsujuifu/pytorch_graph-rel)|
|  [A relational adaptive neural model for joint entity and relation extraction](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.635492/full) |    NYT、WebNLG     |   P,R,Micro-F1  |
|  [Entity-relation extraction as multi-turn question answering](https://aclanthology.org/P19-1129/)  |  ACE04、ACE05、CoNLL04   | P R micro-F1  |[GitHub](https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA)|
|  [Two are better than one: Joint entity and relation extraction with table-sequence encoders](https://aclanthology.org/2020.emnlp-main.133/)| ACE04、ACE05、CoNLL04、ADE  |   P,R,Micro-F1 | [GitHub](https://github.com/LorrinWWW/two-are-better-than-one)|
|  [Joint extraction of entities and overlapping relations using source-target entity labeling](https://www.sciencedirect.com/science/article/abs/pii/S0957417421002943) |   NYT、WebNLG    |  P,R,Micro-F1 |
|  [A multigate encoder for joint entity and relation extraction](https://aclanthology.org/2022.ccl-1.75/) |   ACE04、ACE05、SciERC   | Micro-F1  |
|  [Prta:joint extraction of medical nested entities and overlapping relation via parameter sharing progressive recognition and targeted assignment decoding scheme](https://www.sciencedirect.com/science/article/abs/pii/S0010482524006231) |   NYT、ACE04、 ACE05   |    P,R,Micro-F1   |   
|  [Joint extraction of entities and relations based on a novel tagging scheme](https://aclanthology.org/P17-1113/) |    NYT   |  P,R,Micro-F1  |[GitHub](https://github.com/gswycf/Joint-Extraction-of-Entities-and-Relations-Based-on-a-Novel-Tagging-Scheme)|
|  [Joint extraction of entities and overlapping relations using position-attentive sequence labeling](https://ojs.aaai.org/index.php/AAAI/article/view/4591))|   NYT  | P,R,Micro-F1   |
|  [Joint extraction of entities and relations based on a novel decomposition strategy](https://www.nature.com/articles/s41598-024-51559-w)|    NYT、WebNLG     | P,R,Micro-F1  |[GitHub](https://github.com/yubowen-ph/JointER)|
|  [A novel cascade binary tagging framework for relational triple extraction](https://aclanthology.org/2020.acl-main.136/) |NYT、WebNLG |P,R,Micro-F1  |[GitHub](https://github.com/weizhepei/CasRel)|
|  [Tplinker: Single-stage joint extraction of entities and relations through token pair linking](https://aclanthology.org/2020.coling-main.138/) |   NYT、WebNLG |P,R,Micro-F1  |[GitHub](https://github.com/131250208/TPlinker-joint-extraction)|
|  [PRGC: Potential relation and global correspondence based joint relational triple extraction](https://aclanthology.org/2021.acl-long.486/) |  NYT、WebNLG     | P,R,Micro-F1 |[GitHub](https://github.com/hy-struggle/PRGC)|
|  [A simple overlapping relation extraction method based on dropout](https://ieeexplore.ieee.org/document/9892310) |   NYT、WebNLG    | P,R,Micro-F1  |
|  [Onerel: Joint entity and relation extraction with one module in one step](https://arxiv.org/abs/2203.05412) | NYT、WebNLG  | P,R,Micro-F1 |[GitHub](https://github.com/China-ChallengeHub/OneRel)|
|  [A Novel Tensor Learning Model for Joint Relational Triplet Extraction](https://ieeexplore.ieee.org/document/10109187) | NYT、WebNLG  | P,R,Micro-F1 |
|  [WRTRe: Weighted relative position transformer for joint entity and relation extraction](https://dl.acm.org/doi/10.1016/j.neucom.2021.06.071) | NYT、WebNLG  | P,R,Micro-F1 |

## Relation extraction based on LLMs 
Models targeting only RE tasks.
### Prompt-based Methods
|  Paper  |      Dataset     |Evaluation Metrics  | Code |
| :----- | :--------------: | :---------: |:---------:|
|  [Codeie: Large code generation models are better few-shot information extractors](https://aclanthology.org/2023.acl-long.855.pdf)|   ACE05、CoNLL04、NYT、SciERC    |  Micro-F1 |[GitHub](https://github.com/artpli/CodeIE)|
|  [Codekgc: Code language model for generative knowledge graph construction](https://dl.acm.org/doi/10.1145/3641850) |  ADE、CONLL04、SciERC  | Micro-F1 |[GitHub](https://github.com/zjunlp/DeepKE/blob/main/example/llm/CodeKGC/README_CN.md)|
|  [Retrieval-augmented code generation for universal information extraction](https://link.springer.com/chapter/10.1007/978-981-97-9434-8_3) | ACE05、ADE、CoNLL04、NYT  | Micro-F1|[GitHub](https://github.com/YucanGuo/Code4UIE)|
|  [C-ICL: Contrastive in-context learning for information extraction](https://aclanthology.org/2024.findings-emnlp.590/)| CoNLL04 ACE05 NYT SciERC  |Micro-F1  |
|  [Zero-shot temporal relation extraction with ChatGPT](https://aclanthology.org/2023.bionlp-1.7/) |    TB-Dense、MATRES 、TDDMan | P,R,Micro-F1  |  - |
|  [GPT-RE: in-context learning for relation extraction using large language models](https://aclanthology.org/2023.emnlp-main.214.pdf)| Semeval、TACRED、SciERC、ACE05  |  Micro-F1  |  [GitHub](https://github.com/YukinoWan/GPT-RE)|
|  [Chain of Thought with Explicit Evidence Reasoning for Few-shot Relation Extraction](https://aclanthology.org/2023.emnlp-main.214.pdf)| FewRel   |  Accuracy  |  [GitHub](https://aclanthology.org/2023.findings-emnlp.153.pdf)|
|  [Aligning instruction tasks unlocks large language models as zero-shot relation extractors](https://aclanthology.org/2023.findings-acl.50.pdf) |   TACRED、RETACRED、TACREV、SemEval    |   P,R,Micro-F1 |[GitHub](https://github.com/OSU-NLP-Group/QA4RE)|
|  [Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency](https://aclanthology.org/2024.lrec-main.885/)|REFinD|Micro-F1  |
|  [Revisiting large language models as zero-shot relation extractors](https://aclanthology.org/2023.findings-emnlp.459v2.pdf) |   FewRel、TACRED、TACREV、Re-TACRED|     P,R,Micro-F1  |    
|  [GPT-FinRE: In-context Learning for Financial Relation Extraction using Large Language Models](https://aclanthology.org/2023.finnlp-2.5/) | REFinD|   Micro-F1  |  [GitHub](https://github.com/Upaya07/GPT-FinRE)|
|  [Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models](https://aclanthology.org/2023.emnlp-main.334/) | DocRE|   P,R,Micro-F1  |  [GitHub](https://github.com/bigai-nlco/DocGNRE)|
|  [Construction and Canonicalization of Economic Knowledge Graphs with LLMs](https://link.springer.com/chapter/10.1007/978-3-031-81221-7_23) |   self-build  |  P R micro-F1 |  
|  [Retrieval-Augmented Generation-based Relation Extraction](https://arxiv.org/abs/2404.13397) |TACRED TACREV Re-TACRED SemEval| P,R,Micro-F1  |[GitHub](https://github.com/sefeoglu/RAG4RE)
|  [Event Temporal Relation Extraction based on Retrieval-Augmented on LLMs](https://ieeexplore.ieee.org/document/10651084) |TB-Dense TDDAuto  TDDMan| Micro-F1  |
|  [Document-level Clinical Entity and Relation extraction via Knowledge Base-Guided Generation](https://aclanthology.org/2024.bionlp-1.24/) |n2c2 Dataset、ADE| Micro-F1  |
|  [Construction of a Maritime Knowledge Graph Using GraphRAG for Entity and Relationship Extraction from Maritime Documents](https://www.scirp.org/journal/paperinformation?paperid=140766) |self-build| F1  |
|  [Unlocking instructive in-context learning with tabular prompting for relational triple extraction](https://aclanthology.org/2024.lrec-main.1488/) |CoNLL04、NYT| Micro-F1  |
|  [Document-Level In-Context Few-Shot Relation Extraction via Pre-Trained Language Models](https://arxiv.org/abs/2310.11085)  |DocRED |Micro-F1  |[GiHub](https://github.com/oezyurty/replm)

### Fine-tuning-based Methods
|  Paper  |      Dataset     |Evaluation Metrics    |Code |
| :--------------------- | :---------: | :---------: |:---------: |
|  [Leveraging pre-trained language models for mining microbiome-disease relationships](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05411-z)  |GCS  |P,R,F1   |
|  [Revisiting relation extraction in the era of large language models](https://aclanthology.org/2023.acl-long.868.pdf) |  CONLL04、ADE、NYT  |  Micro-F1  |[GitHub](https://github.com/BenfengXu/S2ynRE)
|  [Does synthetic data generation of llms help clinical text mining?](https://arxiv.org/abs/2303.04360)|  GAD、EU-ADR   |P,R,Micro-F1 |
|  [S2ynRE: Two-stage Self-training with Synthetic data for Low-resource Relation Extraction](https://aclanthology.org/2023.acl-long.455/)|   SemEval  TACRED Re-TACRED  ChemProt |Accuracy,Micro-F1 |
|  [How to unleash the power of large language models for few-shot relation extraction?](https://aclanthology.org/2023.sustainlp-1.13/)|  TACRED TACREV RE-TACRED SciERC  | Micro-F1  |[GitHub](https://github.com/zjunlp/DeepKE/tree/main/example/llm/UnleashLLMRE)|
|  [Empowering Few-Shot Relation Extraction with The Integration of Traditional RE Methods and Large Language Models](https://dl.acm.org/doi/10.1007/978-981-97-5569-1_22)|  TACRED TACREV RE-TACRED  | Micro-F1  |[GitHub](https://github.com/liuyeah/dsare)|
|  [Towards Generating High-Quality Knowledge Graphs by Leveraging Large Language Models](https://link.springer.com/chapter/10.1007/978-3-031-70239-6_31)|  Self-built|Macro-F1 |
|  [Unified structure generation for universal information extraction](https://aclanthology.org/2022.acl-long.395/) |   ACE05、NYT、WebNLG、SciERC    | Micro-F1  |[GitHub](https://github.com/universal-ie/UIE)|
|  [Instructuie: Multi-task instruction tuning for unified information extraction](https://arxiv.org/abs/2304.08085)  |  ACE04、ACE05、CoNLL04、ADE、semeval、GIDS| Micro-F1 |[GitHub](https://github.com/beyonderxx/instructuie)|
|  [REBEL: Relation extraction by end-to-end language generation](https://aclanthology.org/2021.findings-emnlp.204/)| CONLL04、NYT、DocRED、ADE、Re-TACRED| P,R,Micro-F1  | [GitHub](https://github.com/Babelscape/rebel)|
|  [Do instruction-tuned large language models help with relation extraction?](https://ceur-ws.org/Vol-3577/paper15.pdf)  | REBEL |P,R,Micro-F1,Macro-F1  |
|  [AutoRE: Document-level relation extraction with large language models](https://aclanthology.org/2024.acl-demos.20/)  |  Re-DocRED |P,R,Micro-F1   |[GitHub](https://github.com/thudm/autore)|
|  [LLM with Relation Classifier for Document-Level Relation Extraction](https://arxiv.org/abs/2408.13889) |DocRED Re-DocRED  |Micro-F1  |[GiHub](https://github.com/wisper12933/LMRC)
|  [Taiyi: a bilingual fine-tuned large language model for diverse biomedical tasks](https://arxiv.org/abs/2311.11608) |   BC5CDR、CMeIE   | Micro-F1  |[GitHub](https://github.com/dutir-bionlp/taiyi-llm)|   
|  [LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction](https://academic.oup.com/jamia/article-abstract/31/9/2010/7696965?redirectedFrom=fulltext)|Chemprot DDI2013| Micro-F1  |
|  [A Few-Shot Entity Relation Extraction Method in the Legal Domain Based on Large Language Models](https://dl.acm.org/doi/10.1145/3675417.3675513)|CAIL| Micro-F1  |

## Multi-domain relation extraction
|  Paper  |      Dataset   |Evaluation Metrics    |Code |
| :----- | :--------------:| :---------: |:---------: |
|  [Multimodal Relation Extraction with Efficient Graph Alignment](https://dl.acm.org/doi/10.1145/3474085.3476968)|MNRE |P,R,Micro-F1  |[GitHub](https://github.com/thecharm/Mega)
|  [On analyzing the role of image for visual-enhanced relation extraction (student abstract)](https://dl.acm.org/doi/abs/10.1609/aaai.v37i13.26987) |MNRE|Micro-F1  |
|  [Named Entity and Relation Extraction with Multi-Modal Retrieval](https://aclanthology.org/2022.findings-emnlp.437/) |MNRE |Micro-F1  |[GitHub](https://github.com/modelscope/AdaSeq/tree/master/examples/MoRe)
|  [Multimodal Relation Extraction with Cross-Modal Retrieval and Synthesis](https://aclanthology.org/2023.acl-short.27/)  |MNRE |P,R,Micro-F1  |
|  [Good Visual Guidance Make A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction](https://aclanthology.org/2022.findings-naacl.121/) |MNRE|P,R,Micro-F1 |[GitHub](https://github.com/zjunlp/HVPNeT)
|  [Prompt Me Up: Unleashing the Power of Alignments for Multimodal Entity and Relation Extraction](https://dl.acm.org/doi/10.1145/3581783.3611899)  |MNRE |P,R,Micro-F1  |[GitHub](https://github.com/THU-BPM/PROMU)
|  [Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion](https://dl.acm.org/doi/abs/10.1145/3477495.3531992) |MNRE |P,R,Micro-F1  |[GitHub](https://github.com/zjunlp/MKGformer)
|  [Multi-scale interaction network for multimodal entity and relation extraction](https://dl.acm.org/doi/abs/10.1016/j.ins.2024.121787)  |MNRE |P,R,Micro-F1  |
|  [CE-DCVSI: : Multimodal relational extraction based on collaborative enhancement of dual-channel visual semantic information](https://dl.acm.org/doi/10.1016/j.eswa.2024.125608)  |MNRE |P,R,Micro-F1  |
|  [Using Augmented Small Multimodal Models to Guide Large Language Models for Multimodal Relation Extraction](https://www.mdpi.com/2076-3417/13/22/12208)  |MNRE |P,R,Micro-F1  |
|  [UMIE: Unified Multimodal Information Extraction with Instruction Tuning](https://ojs.aaai.org/index.php/AAAI/article/view/29873) |MNRE |P,R,Micro-F1  |[GitHub](https://github.com/ZUCC-AI/UMIE)
|  [Exploiting Visual Relation and Multi-Grained Knowledge for Multimodal Relation Extraction](https://ieeexplore.ieee.org/document/10650770)  |MNRE |P,R,Micro-F1  |
|  [The more quality information the better: : Hierarchical generation of multi-evidence alignment and fusion model for multimodal entity and relation extraction](https://dl.acm.org/doi/10.1016/j.ipm.2024.103875) |MNRE |P,R,Micro-F1  |[GitHub](https://github.com/lsx314/HGMAF)
|  [Caption-Aware Multimodal Relation Extraction with Mutual Information Maximization](https://dl.acm.org/doi/10.1145/3664647.3681219) |MNRE |P,R,Micro-F1  |[GitHub](https://github.com/zefanZhang-cn/CAMIM)


## Relation Extraction Open-source Tools 
Models targeting only EE tasks.
|  Paper  |    Features    | Code |
| :----- | :--------------: | :---------: |
|  [OpenNRE](https://www.mdpi.com/2306-5354/10/5/586)| Support sentence-level, documentlevel, few-shot RE. |[GitHub](https://github.com/thunlp/OpenNRE)|   
|  [OpenUE](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05411-z)  |  Based on PyTorch, supports efficient training and inference. |[GitHub](https://github.com/zjunlp/openue)|   
|  [DeepKE](https://aclanthology.org/2023.acl-long.868.pdf) | Supports cnSchema, low-resource, long-length, and multi-modal RE. |[GitHub](https://github.com/zjunlp/DeepKE)|   
|  [UIE](https://aclanthology.org/2022.acl-long.395/) |   Supports large-scale pre-trained models and fine-tuning on demand.    |[GitHub](https://github.com/universal-ie/UIE)|   

