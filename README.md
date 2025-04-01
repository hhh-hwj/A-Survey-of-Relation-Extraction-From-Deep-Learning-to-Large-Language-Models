
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
|  [1.Relation classification via convolutional deep neural network](https://aclanthology.org/C14-1220/)  |   SemEval-2010 Task 8      | [GitHub](https://github.com/onehaitao/CNN-relation-extraction)|
|  [2.Relation extraction: Perspective from convolutional neural networks](https://aclanthology.org/W15-1506/)  | SemEval-2010 Task 8 dataset  |
|  [3.Classifying relations by ranking with convolutional neural networks](https://aclanthology.org/P15-1061/)  | SemEval-2010 Task 8 dataset  |[GitHub](https://github.com/onehaitao/CR-CNN-relation-extraction)|
|  [4.Relation classification via multi-level attention CNNs](https://aclanthology.org/P16-1123/)  |  SemEval-2010 Task 8   |[GitHub](https://github.com/FrankWork/acnn)|
|  [5.Semantic relation classification via convolutional neural networks with simple negative sampling][(https://aclanthology.org/D15-1062/) |   SemEval-2010 Task 8    |   SemEval-2010 Task 8   |    |
|  [6.Knowledge-oriented convolutional neural network for causal relation extraction from natural language texts](https://www.sciencedirect.com/science/article/abs/pii/S0957417418305177)  |   SemEval-2010 Task 8   ||
|  [7.Semantic compositionality through recursive matrix-vector spaces](https://aclanthology.org/D12-1110/)|   SemEval-2010 Task 8    |      |
|  [8.Simple customization of recursive neural networks for semantic relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）       |
|  [9.Chain based rnn for relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）   |
|  [10.Improved relation classification by deep recurrent neural networks with data augmentation](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）        
|  [11.Classifying relations via long short term memory networks along shortest dependency paths](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）          |
|  [12.Attentionbased bidirectional long short-term memory networks for relation classification](https://aclanthology.org/2023.findings-emnlp.1040/)|   SemEval-2010 Task 8     |[GitHub]（https://github.com/FrankWork/acnn）          |
|  [13.Attention-based lstm with filter mechanism for entity relation classification](https://aclanthology.org/2023.findings-emnlp.1040/)|   SemEval-2010 Task 8     |[GitHub]（https://github.com/FrankWork/acnn）     |
|  [14.Direction-sensitive relation extraction using bi-sdp attention model](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）     |
|  [15.A dependencybased neural network for relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）     |
|  [16.Bidirectional recurrent convolutional neural network for relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）    |
|  [17.Structure regularized neural network for entity relation classification for Chinese literature text](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）    |
|  [18.Neural relation classification with text descriptions](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）    |
|  [19.A combination of rnn and cnn for attention-based relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）   |
|  [20.A single attentionbased combination of cnn and rnn for relation classification](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8    |[GitHub]（https://github.com/FrankWork/acnn）     |
|  [21.Relation classification using segment-level attention-based CNN and dependencybased RNN](https://aclanthology.org/2023.findings-emnlp.1040/) |   SemEval-2010 Task 8     |[GitHub]（https://github.com/FrankWork/acnn）     |
## Joint Extraction-based Methods
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
| [Granular Entity Mapper: Advancing Fine-grained Multimodal Named Entity Recognition and Grounding](https://aclanthology.org/2024.findings-emnlp.183/) | EMNLP Findings | 2024 | |
| [Double-Checker: Large Language Model as a Checker for Few-shot Named Entity Recognition](https://aclanthology.org/2024.findings-emnlp.180/) | EMNLP Findings | 2024 | [GitHub](https://github.com/fanshu6hao/Double-Checker) |
| [VerifiNER: Verification-augmented NER via Knowledge-grounded Reasoning with Large Language Models](https://aclanthology.org/2024.acl-long.134/) | ACL | 2024 | [GitHub](https://github.com/emseoyk/VerifiNER) | 
| [ProgGen: Generating Named Entity Recognition Datasets Step-by-step with Self-Reflexive Large Language Models](https://aclanthology.org/2024.findings-acl.947/) | ACL Findings | 2024 | [GitHub](https://github.com/StefanHeng/ProgGen) |
| [Rethinking Negative Instances for Generative Named Entity Recognition](https://aclanthology.org/2024.findings-acl.206/) |  ACL Findings | 2024 | [GitHub](https://github.com/yyDing1/GNER) |
| [LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition](https://aclanthology.org/2024.findings-acl.76/) | ACL Findings | 2024 | [GitHub](https://github.com/JinYuanLi0012/RiVEG) |
| [RT: a Retrieving and Chain-of-Thought framework for few-shot medical named entity recognition](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocae095/7665312) | Others | 2024-05 | [GitHub](https://github.com/ToneLi/RT-Retrieving-and-Thinking) |
| [P-ICL: Point In-Context Learning for Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2405.04960) | Arxiv | 2024-06 | [GitHub](https://github.com/jiangguochaoGG/P-ICL) |
| [Astro-NER -- Astronomy Named Entity Recognition: Is GPT a Good Domain Expert Annotator?](https://arxiv.org/abs/2405.02602) | Arxiv | 2024-05 | []() |
| [Know-Adapter: Towards Knowledge-Aware Parameter-Efficient Transfer Learning for Few-shot Named Entity Recognition](https://aclanthology.org/2024.lrec-main.854/) | COLING | 2024 | []() |
| [ToNER: Type-oriented Named Entity Recognition with Generative Language Model](https://aclanthology.org/2024.lrec-main.1412.pdf) | COLING | 2024 | []() |
| [CHisIEC: An Information Extraction Corpus for Ancient Chinese History](https://aclanthology.org/2024.lrec-main.283/)  | COLING | 2024 | [GitHub](https://github.com/tangxuemei1995/CHisIEC) |
| [Astronomical Knowledge Entity Extraction in Astrophysics Journal Articles via Large Language Models](https://iopscience.iop.org/article/10.1088/1674-4527/ad3d15/meta) | Others | 2024-04 | []() |
| [LTNER: Large Language Model Tagging for Named Entity Recognition with Contextualized Entity Marking](https://arxiv.org/abs/2404.05624) | Arxiv | 2024-04 | [GitHub](https://github.com/YFR718/LTNER) |
| [Enhancing Software-Related Information Extraction via Single-Choice Question Answering with Large Language Models](https://arxiv.org/pdf/2404.05587)  | Others | 2024-04 | []() |
| [Knowledge-Enriched Prompt for Low-Resource Named Entity Recognition](https://dl.acm.org/doi/abs/10.1145/3659948) | TALLIP | 2024-04 | []() |
| [VANER: Leveraging Large Language Model for Versatile and Adaptive Biomedical Named Entity Recognition](https://arxiv.org/abs/2404.17835) | Arxiv | 2024-04 | [GitHub](https://github.com/Eulring/VANER) |
| [LLMs in Biomedicine: A study on clinical Named Entity Recognition](https://arxiv.org/pdf/2404.07376) | Arxiv | 2024-04 | []() |
| [Out of Sesame Street: A Study of Portuguese Legal Named Entity Recognition Through In-Context Learning](https://www.researchgate.net/profile/Rafael-Nunes-35/publication/379665297_Out_of_Sesame_Street_A_Study_of_Portuguese_Legal_Named_Entity_Recognition_Through_In-Context_Learning/links/6614701839e7641c0ba6879b/Out-of-Sesame-Street-A-Study-of-Portuguese-Legal-Named-Entity-Recognition-Through-In-Context-Learning.pdf) | ResearchGate | 2024-04 | []() |
| [Mining experimental data from Materials Science literature with Large Language Models: an evaluation study](https://arxiv.org/abs/2401.11052) | Arxiv | 2024-04 | [GitHub](https://github.com/lfoppiano/MatSci-LumEn) |
| [LinkNER: Linking Local Named Entity Recognition Models to Large Language Models using Uncertainty](https://arxiv.org/abs/2402.10573) | WWW | 2024 | 
| [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://aclanthology.org/2024.naacl-short.49/)   |   NAACL Short |  2024    | [GitHub](https://github.com/Emma1066/Self-Improve-Zero-Shot-NER) |
| [On-the-fly Definition Augmentation of LLMs for Biomedical NER](https://arxiv.org/abs/2404.00152) | NAACL | 2024 | [GitHub](https://github.com/allenai/beacon) |
| [MetaIE: Distilling a Meta Model from LLM for All Kinds of Information Extraction Tasks](https://arxiv.org/abs/2404.00457)  | Arxiv | 2024-03 | [GitHub](https://github.com/KomeijiForce/MetaIE) |
| [Distilling Named Entity Recognition Models for Endangered Species from Large Language Models](https://arxiv.org/abs/2403.15430)  | Arxiv | 2024-03 | []() |
| [Augmenting NER Datasets with LLMs: Towards Automated and Refined Annotation](https://arxiv.org/abs/2404.01334) | Arxiv | 2024-03 | []() |
| [ConsistNER: Towards Instructive NER Demonstrations for LLMs with the Consistency of Ontology and Context](https://ojs.aaai.org/index.php/AAAI/article/view/29892)| AAAI | 2024 | 
| [Embedded Named Entity Recognition using Probing Classifiers](https://arxiv.org/abs/2403.11747) | Arxiv | 2024-03 | [GitHub](https://github.com/nicpopovic/EMBER) |
| [In-Context Learning for Few-Shot Nested Named Entity Recognition](https://arxiv.org/abs/2402.01182) | Arxiv | 2024-02 | []() |
| [LLM-DA: Data Augmentation via Large Language Models for Few-Shot Named Entity Recognition](https://arxiv.org/abs/2402.14568) | Arxiv | 2024-02 | []() |
| [Structured information extraction from scientific text with large language models](https://www.nature.com/articles/s41467-024-45563-x) | Nature Communications | 2024-02 | [GitHub](https://github.com/lbnlp/nerre-llama) |
| [NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data](https://arxiv.org/abs/2402.15343) |  Arxiv | 2024-02 | 
| [A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction](https://arxiv.org/abs/2402.13364) | Arxiv | 2024-02 | 
| [PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition](https://arxiv.org/abs/2402.04838) | Arxiv | 2024-02 |
| [Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction](https://arxiv.org/abs/2402.14373) | Arxiv | 2024-02 | |
| [C-ICL: Contrastive In-context Learning for Information Extraction](https://arxiv.org/abs/2402.11254) | Arxiv | 2024-02 | 
|  [UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition](https://openreview.net/pdf?id=r65xfUb76p)  |   ICLR    |  2024   |  [GitHub](https://github.com/universal-ner/universal-ner)  |
| [Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering](https://arxiv.org/abs/2303.16416v3) | Arxiv | 2024-01 | [GitHub](https://github.com/BIDS-Xu-Lab/Clinical_Entity_Recognition_Using_GPT_models) |
|  [2INER: Instructive and In-Context Learning on Few-Shot Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.259/)  |   EMNLP Findings    |  2023-12   |    |
|  [In-context Learning for Few-shot Multimodal Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.196/)  |   EMNLP Findings    |  2023-12   |    |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |   EMNLP Findings    |  2023-12   |  [GitHub](https://github.com/mayubo2333/LLM-IE)  |
|  [Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset](https://arxiv.org/abs/2310.10118)  |   EMNLP     |  2023-12   |  [GitHub](https://github.com/CompNet/conivel/tree/gen)  |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings    |  2023-12   |  [GitHub](https://github.com/ridiculouz/LLMAAA)  |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition with Auxiliary Refined Knowledge](https://arxiv.org/abs/2305.12212)  |   EMNLP Findings    |  2023-12   |  [GitHub](https://github.com/JinYuanLi0012/PGIM)  |
| [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526) | Arxiv | 2023-11 | [GitHub](https://github.com/urchade/GLiNER) |
| [GPT Struct Me: Probing GPT Models on Narrative Entity Extraction](https://ieeexplore.ieee.org/abstract/document/10350182) | WI-IAT | 2023-10 | [GitHub](https://github.com/hmosousa/gpt_struct_me) |
|  [GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/abs/2304.10428)  |   Arxiv    |  2023-10   |  [GitHub](https://github.com/ShuheWang1998/GPT-NER)  |
|  [Prompt-NER: Zero-shot Named Entity Recognition in Astronomy Literature via Large Language Models](https://arxiv.org/abs/2310.17892)  |   Arxiv    |  2023-10   |    |
|  [Inspire the Large Language Model by External Knowledge on BioMedical Named Entity Recognition](https://arxiv.org/abs/2309.12278)  |   Arxiv    |  2023-09   |    |
|  [One Model for All Domains: Collaborative Domain-Prefx Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)  |   IJCAI    |  2023-09   |  [GitHub](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross)  |
|  [Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction](https://arxiv.org/abs/2306.14122)  |   Arxiv    |  2023-08   |    |
| [Learning In-context Learning for Named Entity Recognition ](https://aclanthology.org/2023.acl-long.764/) | ACL | 2023-07 | [GitHub](https://github.com/chen700564/metaner-icl) |
|  [Debiasing Generative Named Entity Recognition by Calibrating Sequence Likelihood](https://aclanthology.org/2023.acl-short.98/)  |   ACL Short    |  2023-07   |    |
|  [Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks](https://aclanthology.org/2023.findings-acl.578/)  |   ACL Findings     |  2023-07   |    |
|  [Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction](https://aclanthology.org/2023.bionlp-1.15/)  |   BioNLP    |  2023-07   |  [GitHub](https://github.com/arkhn/bio-nlp2023)  |
| [NAG-NER: a Unified Non-Autoregressive Generation Framework for Various NER Tasks](https://aclanthology.org/2023.acl-industry.65/) | ACL Industry | 2023-07 | 
| [Unified Named Entity Recognition as Multi-Label Sequence Generation](https://ieeexplore.ieee.org/abstract/document/10191921) |  IJCNN | 2023-06 | 
|  [PromptNER : Prompting For Named Entity Recognition](https://arxiv.org/abs/2305.15444)  |   Arxiv    |  2023-06   |    |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Arxiv    |  2023-04   |    |
| [Unified Text Structuralization with Instruction-tuned Language Models](https://arxiv.org/abs/2303.14956)  | Arxiv | 2023-03 | []() |
|  [Structured information extraction from complex scientific text with fine-tuned large language models](https://arxiv.org/abs/2212.05238)  |   Arxiv    |  2022-12   |  [Demo](http://www.matscholar.com/info-extraction)  |
|  [LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting](https://aclanthology.org/2022.coling-1.209/) | COLING |  2022-10 |  [GitHub](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot) |
|  [De-bias for generative extraction in unified NER task](https://aclanthology.org/2022.acl-long.59.pdf)  |   ACL      |  2022-05   |    |
|  [InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER](https://arxiv.org/abs/2203.03903) |  Arxiv  |  2022-03 |   |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |   EMNLP      |  2021-11   |  [GitHub](https://github.com/PlusLabNLP/TempGen)  |
|  [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)  |   ACL      |  2021-08   |  [GitHub](https://github.com/yhcc/BARTNER)  |
|  [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.findings-acl.161.pdf)  |   ACL Findings    |  2021-08   |  [GitHub](https://github.com/Nealcly/templateNER)  |

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
