# Language-Modeling-for-Urban-Mobility-A-Data-Centric-Survey
> Language Modeling for Urban Mobility: A Data-Centric Review and Guidelines

![01-Data](assets/01-introduction.png)


> This repository contains a collection of resources and papers on applying language modeling paradigms to urban mobility scenarios.

![00-intro](assets/00-intro.png)

we propose a comprehensive and data-centric survey of language modeling for urban mobility, structured along three key dimensions: 
- **(i) How to transform heterogeneous mobility data into language model–like formats through tokenization, encoding, and prompting;** 
- **(ii) How to choose among different categories of language models, ranging from pretrained language models, large language models (LLMs), MLLMs, and diffusion models;**
- **(iii) What are the advantages of applying language modeling to urban mobility in diverse urban downstream tasks?**

## Contents

- [Related Survey](#Related-Survey)
- [1. Discrete Mobility Sequence](#1-discrete-mobility-sequence)
  - [1.1 Tokenization](#11-tokenization)
    - [1.1.2 Pretrained LM](#112-pretrained-lm)
    - [1.1.3 LLM](#113-llm)
  - [1.2 Encoding](#12-encoding)
    - [1.2.1 Attention-based](#121-attention-based)
    - [1.2.2 Pretrained LM](#122-pretrained-lm)
    - [1.2.3 LLM](#123-llm)
    - [1.2.4 Diffusion Model](#124-diffusion-model)
  - [1.3 Prompting](#13-prompting)
    - [1.3.1 LLM](#131-llm)
- [2. Continuous Mobility Sequence](#2-continuous-mobility-sequence)
  - [2.1 Discrete Tokenization (Quantization)](#21-discrete-tokenization-quantization)
    - [2.1.1 Pretrained LM](#211-pretrained-lm)
    - [2.1.2 LLM](#212-llm)
  - [2.2 Encoding](#22-encoding)
    - [2.2.1 Pretrained LM](#221-pretrained-lm)
    - [2.2.2 LLM](#222-llm)
  - [2.3 Prompting](#23-prompting)
    - [2.3.1 LLM](#231-llm)
  - [2.4 Featurization](#24-featurization)
    - [2.4.1 Diffusion Model](#241-diffusion-model)
- [3. Graph-type Mobility](#3-Graph-type-Mobility)
  - [3.1 Tokenization](#31-tokenization)
    - [3.1.1 Pretrained LM](#311-pretrained-lm)
    - [3.1.2 LLM](#312-llm)
  - [3.2 Encoding](#32-encoding)
    - [3.2.1 Pretrained LM](#321-pretrained-lm)
    - [3.2.2 LLM](#322-llm)
  - [3.3 Prompting](#33-prompting)
    - [3.3.1 LLM](#331-llm)
  - [3.4 Featurization](#34-featurization)
    - [3.4.1 Diffusion Model](#341-diffusion-model)
- [4. Multimodal Mobility Data](#4-multimodal-mobility-data)
  - [4.1 Vision and Trajectory](#41-vision-and-trajectory)
  - [4.2 Text and Trajectory](#42-text-and-trajectory)
  - [4.3 Vision and Traffic](#43-vision-and-traffic)
  - [4.4 Text and Traffic](#44-text-and-traffic)
  - [4.5 Vision and Graph](#45-vision-and-graph)
  - [4.6 Text and Graph](#46-text-and-graph)

- [Dataset](#Dataset)
- [References from other AI Communities](#References-from-Other-AI-Communities)

# Related Survey



## 1. Discrete Mobility Sequence

![02-discrete](assets/02-discrete-modeling.png)


### 1.1 Tokenization

#### 1.1.1 Pretrained LM

- **Attention**
  - MoveSim: Learning to simulate human mobility (`feng2020learning`) KDD, 2020. [[Link]](TBD)
  - TPG (`luo2023timestamps`) Timestamps as prompts for geography-aware location recommendation (`luo2023timestamps`) CIKM, 2023. [[PDF]](TBD) [[Link]](TBD)

- **Masked LM**
  - CTLE (`lin2021pre`) Pre-training context and time aware location embeddings from spatial-temporal trajectories for user next location prediction (`lin2021pre`) AAAI, 2021. [[PDF]](TBD) [[Link]](TBD)
  - Wepos (`guo2022wepos`) Wepos: Weak-supervised indoor positioning with unlabeled wifi for on-demand delivery (`guo2022wepos`) IMWUT, 2022. [[PDF]](TBD) [[Link]](TBD)
  - Yang et al. (`yang2024applying`) Applying masked language model for transport mode choice behavior prediction (`yang2024applying`) TR-A, 2024. [[PDF]](TBD) [[Link]](TBD)
  - GREEN (`zhou2025grid`) Grid and road expressions are complementary for trajectory representation learning (`zhou2025grid`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
  - TrajBERT (`trajbert2023`) TrajBERT: BERT-Based Trajectory Recovery with Spatial-Temporal Refinement (`trajbert2023`) TMC, 2023. [[PDF]](TBD) [[Link]](https://doi.org/10.1109/TMC.2023.3297115)

- **Transformer Decoder**
  - MobilityGPT (`mobilitygpt2024`)
  - GeoFormer (`solatorio2023geoformer`) GeoFormer: predicting human mobility using generative pre-trained transformer (GPT) (`solatorio2023geoformer`) 1st International Workshop on the Human, 2023. [[PDF]](TBD) [[Link]](TBD)
  - LMTAD (`mbuya2024trajectory`) Trajectory Anomaly Detection with Language Models (`mbuya2024trajectory`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Kobayashi et al. (`kobayashi2023modeling`) Modeling and generating human mobility trajectories using transformer with day encoding (`kobayashi2023modeling`) 1st International Workshop on the Human, 2023. [[PDF]](TBD) [[Link]](TBD)
  - Traj-LLM (`lan2024traj`) Traj-llm: A new exploration for empowering trajectory prediction with pre-trained large language models (`lan2024traj`) IEEE Transactions on Intelligent Vehicle, 2024. [[PDF]](TBD) [[Link]](TBD)
  - TrajLearn (`nadiri2025trajlearn`) TrajLearn: Trajectory Prediction Learning using Deep Generative Models (`nadiri2025trajlearn`) ACM Transactions on Spatial Algorithms a, 2025. [[PDF]](TBD) [[Link]](TBD)

#### 1.1.2 LLM
- GNPR-SID (`wang2025generativekdd25`) Generative Next POI Recommendation with Semantic ID (`wang2025generativekdd25`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
- RHYTHM (`he2025rhythm`) RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility (`he2025rhythm`) arXiv preprint arXiv:2509.23115, 2025. [[PDF]](TBD) [[Link]](TBD)
- MobGLM (`zhang2024mobglm`) MobGLM: A Large Language Model for Synthetic Human Mobility Generation (`zhang2024mobglm`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
- MobilityGPT (`mobilitygpt2024`)
- Geo-Llama (`li2024geo`) Geo-llama: Leveraging llms for human mobility trajectory generation with spatiotemporal constraints (`li2024geo`) arXiv preprint arXiv:2408.13918, 2024. [[PDF]](TBD) [[Link]](TBD)




### 1.2 Encoding

#### 1.2.1 Attention-based
- Deepmove (`feng2018deepmove`) Deepmove: Predicting human mobility with attentional recurrent networks (`feng2018deepmove`) WWW, 2018. [[PDF]](TBD) [[Link]](TBD)
- STRNN (`liu2016predicting`) Predicting the next location: A recurrent model with spatial and temporal contexts (`liu2016predicting`) AAAI, 2016. [[PDF]](TBD) [[Link]](TBD)
- LSTPM (`sun2020go`) Where to go next: Modeling long-and short-term user preferences for point-of-interest recommendation (`sun2020go`) AAAI, 2020. [[PDF]](TBD) [[Link]](TBD)
- STAN (`Luo2021stan`) STAN: Spatio-Temporal Attention Network for Next Location Recommendation (`Luo2021stan`) WWW, 2021. [[PDF]](TBD) [[Link]](https://doi.org/10.1145/3442381.3449998)

#### 1.2.2 Pretrained LM

- **Masked LM**
  - LP-BERT (`suzuki2024cross`) Cross-city-aware Spatiotemporal BERT (`suzuki2024cross`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
  - TraceBERT (`crivellari2022tracebert`) Tracebert—a feasibility study on reconstructing spatial--temporal gaps from incomplete motion trajectories via bert training process on discrete location sequences (`crivellari2022tracebert`) Sensors, 2022. [[PDF]](TBD) [[Link]](TBD)
  - CTLE (`lin2021pre`)
  - GREEN (`zhou2025grid`)

- **Causal Attention**
  - MobTCast (`xue2021mobtcast`) MobTCast: Leveraging auxiliary trajectory forecasting for human mobility prediction (`xue2021mobtcast`) NeurIPS, 2021. [[PDF]](TBD) [[Link]](TBD)
  - MoveSim (`feng2020learning`)
  - AttnMove (`xia2021attnmove`) Attnmove: History enhanced trajectory recovery via attentional network (`xia2021attnmove`) AAAI, 2021. [[PDF]](TBD) [[Link]](TBD)

#### 1.2.3 LLM
- LLMEmb (`liu2025llmemb`) Llmemb: Large language model can be a good embedding generator for sequential recommendation (`liu2025llmemb`) AAAI, 2025. [[PDF]](TBD) [[Link]](TBD)
- Mobility-LLM (`mobilityllm2024`) Mobility-llm: Learning visiting intentions and travel preference from human mobility data with large language models (`mobilityllm2024`) NeurIPS, 2024. [[PDF]](TBD) [[Link]](TBD)
- NextLocLLM (`nextlocllm2025`)
- GSTM-HMU (`luo2025gstm`) GSTM-HMU: Generative Spatio-Temporal Modeling for Human Mobility Understanding (`luo2025gstm`) arXiv preprint arXiv:2509.19135, 2025. [[PDF]](TBD) [[Link]](TBD)

#### 1.2.4 Diffusion Model
- Cardiff (`guo2025leveraging`) Leveraging the Spatial Hierarchy: Coarse-to-fine Trajectory Generation via Cascaded Hybrid Diffusion (`guo2025leveraging`) arXiv preprint arXiv:2507.13366, 2025. [[PDF]](TBD) [[Link]](TBD)
- Diff-POI (`qin2023diffusion`) A diffusion model for poi recommendation (`qin2023diffusion`) ACM Transactions on Information Systems, 2023. [[PDF]](TBD) [[Link]](TBD)
- AutoSTDiff (`xu2025autostdiff`) AutoSTDiff: Autoregressive Spatio-Temporal Denoising Diffusion Model for Asynchronous Trajectory Generation (`xu2025autostdiff`) SIAM, 2025. [[PDF]](TBD) [[Link]](TBD)
- DiffMove (`long2025diffmove`) DiffMove: Group Mobility Tendency Enhanced Trajectory Recovery via Diffusion Model (`long2025diffmove`) arXiv preprint arXiv:2503.18302, 2025. [[PDF]](TBD) [[Link]](TBD)
- GenMove (`long2025one`) One Fits All: General Mobility Trajectory Modeling via Masked Conditional Diffusion (`long2025one`) arXiv preprint arXiv:2501.13347, 2025. [[PDF]](TBD) [[Link]](TBD)
- Diff-DGMN (`zuo2024diff`) Diff-dgmn: A diffusion-based dual graph multi-attention network for poi recommendation (`zuo2024diff`) IEEE Internet of Things Journal, 2024. [[PDF]](TBD) [[Link]](TBD)
- DCPR (`long2024diffusion`) Diffusion-based cloud-edge-device collaborative learning for next POI recommendations (`long2024diffusion`) KDD, 2024. [[PDF]](TBD) [[Link]](TBD)
- Traveller (`luo2025traveller`) Traveller: Travel-Pattern Aware Trajectory Generation via Autoregressive Diffusion Models (`luo2025traveller`) Information Fusion, 2025. [[PDF]](TBD) [[Link]](TBD)
- TrajGDM (`chu2024simulating`) Simulating human mobility with a trajectory generation framework based on diffusion model (`chu2024simulating`) International Journal of Geographical In, 2024. [[PDF]](TBD) [[Link]](TBD)


### 1.3 Prompting
![03-prompt](assets/03-discrete-prompt.png)

#### 1.3.1 LLM

- **As Representor**
  - Poi-enhancer (`cheng2025poi`) Poi-enhancer: An llm-based semantic enhancement framework for poi representation learning (`cheng2025poi`) AAAI, 2025. [[PDF]](TBD) [[Link]](TBD)
  - LLM-Mob (`wang2023would`) Where would i go next? large language models as human mobility predictors (`wang2023would`) arXiv preprint arXiv:2308.15197, 2023. [[PDF]](TBD) [[Link]](TBD)
  - TrajCogn (`zhou2024trajcogn`) TrajCogn: Leveraging LLMs for Cognizing Movement Patterns and Travel Purposes from Trajectories (`zhou2024trajcogn`) arXiv preprint arXiv:2405.12459, 2024. [[PDF]](TBD) [[Link]](TBD)

- **As Predictor**
  - TPP-LLM (`liu2024tpp`) Tpp-llm: Modeling temporal point processes by efficiently fine-tuning large language models (`liu2024tpp`) arXiv preprint arXiv:2410.02062, 2024. [[PDF]](TBD) [[Link]](TBD)
  - CoMaPOI (`zhong2025comapoi`) CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language (`zhong2025comapoi`) SIGIR, 2025. [[PDF]](TBD) [[Link]](TBD)
  - AgentMove (`feng2024agentmove`) Agentmove: A large language model based agentic framework for zero-shot next location prediction (`feng2024agentmove`) arXiv preprint arXiv:2408.13986, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Feng et al. (`feng2024move`) Where to move next: Zero-shot generalization of llms for next poi recommendation (`feng2024move`) 2024 ieee conference on artificial intel, 2024. [[PDF]](TBD) [[Link]](TBD)
  - LLM4Poi (`li2024large`) Large language models for next point-of-interest recommendation (`li2024large`) SIGIR, 2024. [[PDF]](TBD) [[Link]](TBD)
  - CSA-Rec (`wang2025collaborative`) Collaborative Semantics-Assisted Large Language Models for Next POI Recommendation (`wang2025collaborative`) ICASSP, 2025. [[PDF]](TBD) [[Link]](TBD)
  - LAMP (`balsebre2024lamp`) LAMP: A language model on the map (`balsebre2024lamp`) arXiv preprint arXiv:2403.09059, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Zhang et al. (`zhang2023large`) Large Language Models for Spatial Trajectory Patterns Mining.(2023) (`zhang2023large`) arXiv preprint arXiv:2310.04942, 2023. [[PDF]](TBD) [[Link]](TBD)
  - Mo et al. (`mo2023large`) Large language models for travel behavior prediction (`mo2023large`) arXiv preprint arXiv:2312.00819, 2023. [[PDF]](TBD) [[Link]](TBD)
  - POI GPT (`kim2024poi`) POI GPT: Extracting POI information from social media text data (`kim2024poi`) The International Archives of the Photog, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Chen et al. (`chen2025toward`) Toward interactive next location prediction driven by large language models (`chen2025toward`) IEEE Transactions on Computational Socia, 2025. [[PDF]](TBD) [[Link]](TBD)
  - DelayPTC-LLM (`chen2024delayptc`) Delayptc-llm: Metro passenger travel choice prediction under train delays with large language models (`chen2024delayptc`) arXiv preprint arXiv:2410.00052, 2024. [[PDF]](TBD) [[Link]](TBD)

- **Generator**
  - Liu et al. (`liu2025aligning`) Aligning LLM agents with human learning and adjustment behavior: a dual agent approach (`liu2025aligning`) arXiv preprint arXiv:2511.00993, 2025. [[PDF]](TBD) [[Link]](TBD)
  - CoPB (`shao2024chain`) Chain-of-planned-behaviour workflow elicits few-shot mobility generation in llms (`shao2024chain`) arXiv preprint arXiv:2402.09836, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Liu et al. (`liu2023can`) Can language models be used for real-world urban-delivery route optimization? (`liu2023can`) The Innovation, 2023. [[PDF]](TBD) [[Link]](TBD)
  - Bhandari et al. (`bhandari2024urban`) Urban mobility assessment using llms (`bhandari2024urban`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Zheng et al. (`zheng2025urban`) Urban planning in the era of large language models (`zheng2025urban`) Nature computational science, 2025. [[PDF]](TBD) [[Link]](TBD)

- **LLM Agents**
  - LLM-HABG (`meng2025behavior`) Behavior Generation for Heterogeneous Agents in Urban Simulation Deduction: A Multi-Stage Approach Based on Large Language Models (`meng2025behavior`) CCSSTA, 2025. [[PDF]](TBD) [[Link]](TBD)
  - PathGPT (`marcelyn2025pathgpt`) PathGPT: Leveraging Large Language Models for Personalized Route Generation (`marcelyn2025pathgpt`) arXiv preprint arXiv:2504.05846, 2025. [[PDF]](TBD) [[Link]](TBD)
  - LLMTraveler (`wang2024ai`) Ai-driven day-to-day route choice (`wang2024ai`) arXiv preprint arXiv:2412.03338, 2024. [[PDF]](TBD) [[Link]](TBD)
  - GATSim (`liu2025gatsim`) GATSim: Urban Mobility Simulation with Generative Agents (`liu2025gatsim`) arXiv preprint arXiv:2506.23306, 2025. [[PDF]](TBD) [[Link]](TBD)

  - MobAgent (`li2024more`) Be more real: Travel diary generation using llm agents and individual profiles (`li2024more`) arXiv preprint arXiv:2407.18932, 2024. [[PDF]](TBD) [[Link]](TBD)
  - CitySim (`bougie2025citysim`) CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation (`bougie2025citysim`) arXiv preprint arXiv:2506.21805, 2025. [[PDF]](TBD) [[Link]](TBD)
  - TravelPlanner (`xie2024travelplanner`) Travelplanner: A benchmark for real-world planning with language agents (`xie2024travelplanner`) arXiv preprint arXiv:2402.01622, 2024. [[PDF]](TBD) [[Link]](TBD)
  - IDM-GPT (`yang2025independent`) Independent mobility gpt (idm-gpt): A self-supervised multi-agent large language model framework for customized traffic mobility analysis using machine learning models (`yang2025independent`) arXiv preprint arXiv:2502.18652, 2025. [[PDF]](TBD) [[Link]](TBD)



## 2. Continuous Mobility Sequence
![04-continuous](assets/04-continuous.png)
### 2.1 Discrete Tokenization (Quantization)

#### 2.1.1 Pretrained LM

- **Encoder-based (BERT-like)**
  - Giuliari (`giuliari2021transformer`) Transformer networks for trajectory forecasting (`giuliari2021transformer`) ICPR, 2021. [[PDF]](TBD) [[Link]](TBD)
  - BERT4Traj (`yang2025bert4traj`) BERT4Traj: Transformer Based Trajectory Reconstruction for Sparse Mobility Data (`yang2025bert4traj`) arXiv preprint arXiv:2507.03062, 2025. [[PDF]](TBD) [[Link]](TBD)

- **Decoder-based (GPT-like)**
  - MotionLM (`seff2023motionlm`) Motionlm: Multi-agent motion forecasting as language modeling (`seff2023motionlm`) ICCV, 2023. [[PDF]](TBD) [[Link]](TBD)
  - RAW (`zhang2023regions`) Regions are who walk them: a large pre-trained spatiotemporal model based on human mobility for ubiquitous urban sensing (`zhang2023regions`) arXiv preprint arXiv:2311.10471, 2023. [[PDF]](TBD) [[Link]](TBD)

- **Encoder–Decoder-based**
  - UniTraj (`zhu2024unitraj`) Unitraj: Learning a universal trajectory foundation model from billion-scale worldwide traces (`zhu2024unitraj`) arXiv preprint arXiv:2411.03859, 2024. [[PDF]](TBD) [[Link]](TBD)

#### 2.1.2 LLM
- LMTraj (`bae2024can`) Can language beat numerical regression? language-based multimodal trajectory prediction (`bae2024can`) CVPR, 2024. [[PDF]](TBD) [[Link]](TBD)
- RouteLLM (`hallgarten2025routellm`) RouteLLM: A Large Language Model with Native Route Context Understanding to Enable Context-Aware Reasoning (`hallgarten2025routellm`) IMWUT, 2025. [[PDF]](TBD) [[Link]](TBD)
- QT-Mob (`chen2025enhancing`) Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization (`chen2025enhancing`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
- CAMS (`du2025cams`) CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation (`du2025cams`) arXiv preprint arXiv:2506.13599, 2025. [[PDF]](TBD) [[Link]](TBD)
- AutoTimes (`liu2024autotimes`)  <!-- traffic time series --> Autotimes: Autoregressive time series forecasters via large language models (`liu2024autotimes`) NeurIPS, 2024. [[PDF]](TBD) [[Link]](TBD)


### 2.2 Encoding

#### 2.2.1 Pretrained LM
- BERT4Traj (`yang2025bert4traj`)
- EETG-SVAE (`zhang2025end`) End-to-end Trajectory Generation-Contrasting Deep Generative Models and Language Models (`zhang2025end`) ACM Transactions on Spatial Algorithms a, 2025. [[PDF]](TBD) [[Link]](TBD)
- Musleh et al. (`musleh2022towards`) Towards a unified deep model for trajectory analysis (`musleh2022towards`) SIGSPATIAL, 2022. [[PDF]](TBD) [[Link]](TBD)
- UrbanGPT (`li2024urbangpt`) Urbangpt: Spatio-temporal large language models (`li2024urbangpt`) KDD, 2024. [[PDF]](TBD) [[Link]](TBD)
- UniST (`yuan2024unist`) Unist: A prompt-empowered universal model for urban spatio-temporal prediction (`yuan2024unist`) KDD, 2024. [[PDF]](TBD) [[Link]](TBD)
- FlashST (`li2024flashst`) Flashst: A simple and universal prompt-tuning framework for traffic prediction (`li2024flashst`) arXiv preprint arXiv:2405.17898, 2024. [[PDF]](TBD) [[Link]](TBD)
- Traffic-Twitter Transformer (`tsai2022traffic`) Traffic-twitter transformer: A nature language processing-joined framework for network-wide traffic forecasting (`tsai2022traffic`) arXiv preprint arXiv:2206.11078, 2022. [[PDF]](TBD) [[Link]](TBD)
- FlowDistill (`yu2025flowdistill`) FlowDistill: Scalable Traffic Flow Prediction via Distillation from LLMs (`yu2025flowdistill`) arXiv preprint arXiv:2504.02094, 2025. [[PDF]](TBD) [[Link]](TBD)
- Cao et al. (`cao2021bert`) BERT-based deep spatial-temporal network for taxi demand prediction (`cao2021bert`) T-ITS, 2021. [[PDF]](TBD) [[Link]](TBD)
- Ma et al. (`ma2025urban`) Urban rail transit passenger flow prediction using large language model under multi-source spatiotemporal data fusion (`ma2025urban`) Physica A: Statistical Mechanics and its, 2025. [[PDF]](TBD) [[Link]](TBD)
- TrafficBERT (`jin2021trafficbert`) TrafficBERT: Pre-trained model with large-scale data for long-range traffic flow forecasting (`jin2021trafficbert`) Expert Systems with Applications, 2021. [[PDF]](TBD) [[Link]](TBD)
- ST-LLM+ (`liu2025st`) ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction (`liu2025st`) IEEE Transactions on Knowledge and Data, 2025. [[PDF]](TBD) [[Link]](TBD)
- MDTI (`liu2025multimodal`) Multimodal Trajectory Representation Learning for Travel Time Estimation (`liu2025multimodal`) arXiv preprint arXiv:2510.05840, 2025. [[PDF]](TBD) [[Link]](TBD)

#### 2.2.2 LLM
- TPLLM (`ren2024tpllm`) TPLLM: A traffic prediction framework based on pretrained large language models (`ren2024tpllm`) arXiv preprint arXiv:2403.02221, 2024. [[PDF]](TBD) [[Link]](TBD)
- LLM-TFP (`cheng2025llm`) LLM-TFP: Integrating large language models with spatio-temporal features for urban traffic flow prediction (`cheng2025llm`) Applied Soft Computing, 2025. [[PDF]](TBD) [[Link]](TBD)
- NextLocLLM (`nextlocllm2025`)
- Liao et al. (`liao2025next`) Next-Generation Travel Demand Modeling with a Generative Framework for Household Activity Coordination (`liao2025next`) arXiv preprint arXiv:2507.08871, 2025. [[PDF]](TBD) [[Link]](TBD)


### 2.3 Prompting

#### 2.3.1 LLM
- **Representation and Mining**
  - Zhang et al. (`zhang2024large`) Large language models for spatial trajectory patterns mining (`zhang2024large`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
  - GPT-J (`ji2024evaluating`) Evaluating the Effectiveness of Large Language Models in Representing and Understanding Movement Trajectories (`ji2024evaluating`) arXiv preprint arXiv:2409.00335, 2024. [[PDF]](TBD) [[Link]](TBD)
  - GeoLLM (`manvi2023geollm`) Geollm: Extracting geospatial knowledge from large language models (`manvi2023geollm`) arXiv preprint arXiv:2310.06213, 2023. [[PDF]](TBD) [[Link]](TBD)
  - AuxMobLCast (`xue2022leveraging`) Leveraging language foundation models for human mobility forecasting (`xue2022leveraging`) SIGSPATIAL, 2022. [[PDF]](TBD) [[Link]](TBD)
  - Wang et al. (`wang2025event`) Event-aware analysis of cross-city visitor flows using large language models and social media data (`wang2025event`) arXiv preprint arXiv:2505.03847, 2025. [[PDF]](TBD) [[Link]](TBD)

- **Prediction**
  - LLM-MPE (`liang2024exploring`) Exploring large language models for human mobility prediction under public events (`liang2024exploring`) Computers, Environment and Urban Systems, 2024. [[PDF]](TBD) [[Link]](TBD)
  - STCInterLLM (`li2025causal`) Causal Intervention Is What Large Language Models Need for Spatio-Temporal Forecasting (`li2025causal`) IEEE Transactions on Cybernetics, 2025. [[PDF]](TBD) [[Link]](TBD)
  - xTP-LLM (`guo2024towards`) Towards explainable traffic flow prediction with large language models (`guo2024towards`) Communications in Transportation Researc, 2024. [[PDF]](TBD) [[Link]](TBD)
  - Cai et al. (`cai2024temporal`) Temporal-Spatial Traffic Flow Prediction Model Based on Prompt Learning (`cai2024temporal`) ISPRS, 2024. [[PDF]](TBD) [[Link]](TBD)
  - LLM4PT (`wu2025llm4pt`) LLM4PT: A large language model-based system for flexible and explainable public transit demand prediction (`wu2025llm4pt`) Computers & Industrial Engineering, 2025. [[PDF]](TBD) [[Link]](TBD)
  - TransLLM (`leng2025transllm`) TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting (`leng2025transllm`) arXiv preprint arXiv:2508.14782, 2025. [[PDF]](TBD) [[Link]](TBD)

- **Generation**
  - LLMob (`jiawei2024large`) Large language models as urban residents: An llm agent framework for personal mobility generation (`jiawei2024large`) NeurIPS, 2024. [[PDF]](TBD) [[Link]](TBD)


### 2.4 Featurization

#### 2.4.1 Diffusion Model
- CoDiffMob (`codiffmob2025`) - Noise Matters: Diffusion Model-Based Urban Mobility Generation with Collaborative Noise Priors (`codiffmob2025`) WWW, 2025. [[PDF]](TBD) [[Link]](TBD)
- ControlTraj (`zhu2024controltraj`) - Controltraj: Controllable trajectory generation with topology-constrained diffusion model (`zhu2024controltraj`) KDD, 2024. [[PDF]](TBD) [[Link]](TBD)
- DiffTraj (`difftraj2023`) - DiffTraj: Generating GPS Trajectories with Diffusion Probabilistic Models (`difftraj2023`) NeurIPS, 2023. [[PDF]](TBD) [[Link]](TBD)
- Cardiff (`guo2025leveraging`)
- UniMob (`long2025universal`) - A universal model for human mobility prediction (`long2025universal`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)



## 3. Graph-type Mobility

![05-graph](assets/05-graph.png)

### 3.1 Tokenization

#### 3.1.1 Pretrained LM
- UniFlow (`yuan2024uniflow`) UniFlow: A Foundation Model for Unified Urban Spatio-Temporal Flow Prediction (`yuan2024uniflow`) arXiv preprint arXiv:2411.12972, 2024. [[PDF]](TBD) [[Link]](TBD)
- RePST (`wang2024repst`) RePST: Language Model Empowered Spatio-Temporal Forecasting via Semantic-Oriented Reprogramming (`wang2024repst`) arXiv preprint arXiv:2408.14505, 2024. [[PDF]](TBD) [[Link]](TBD)
- CompactST (`han2025scalable`) Scalable Pre-Training of Compact Urban Spatio-Temporal Predictive Models on Large-Scale Multi-Domain Data (`han2025scalable`) VLDB, 2025. [[PDF]](TBD) [[Link]](TBD)
- STD-PLM (`huang2025std`) - Std-plm: Understanding both spatial and temporal properties of spatial-temporal data with plm (`huang2025std`) AAAI, 2025. [[PDF]](TBD) [[Link]](TBD)


#### 3.1.2 LLM
- STG-LLM (`liu2024can`) Can large language models capture human travel behavior? evidence and insights on mode choice (`liu2024can`) Evidence and Insights on Mode Choice (Au, 2024. [[PDF]](TBD) [[Link]](TBD)
- ST-LLM (`liu2024spatial`) Spatial-temporal large language model for traffic prediction (`liu2024spatial`) MDM, 2024. [[PDF]](TBD) [[Link]](TBD)



### 3.2 Encoding

#### 3.2.1 Pretrained LM
- STGormer (`zhou2024navigating`) Navigating spatio-temporal heterogeneity: A graph transformer approach for traffic forecasting (`zhou2024navigating`) arXiv preprint arXiv:2408.10822, 2024. [[PDF]](TBD) [[Link]](TBD)
- STGLLM-E (`rong2024edge`) Edge computing enabled large-scale traffic flow prediction with GPT in intelligent autonomous transport system for 6G network (`rong2024edge`) T-ITS, 2024. [[PDF]](TBD) [[Link]](TBD)
- CityCAN (`wang2024citycan`) - CityCAN: Causal attention network for citywide spatio-temporal forecasting (`wang2024citycan`) WSDM, 2024. [[PDF]](TBD) [[Link]](TBD)
- STTNs (`xu2020spatial`) - Spatial-temporal transformer networks for traffic flow forecasting (`xu2020spatial`) arXiv preprint arXiv:2001.02908, 2020. [[PDF]](TBD) [[Link]](TBD)
- ST-LINK (`jeon2025st`) - ST-LINK: Spatially-Aware Large Language Models for Spatio-Temporal Forecasting (`jeon2025st`) arXiv preprint arXiv:2509.13753, 2025. [[PDF]](TBD) [[Link]](TBD)

#### 3.2.2 LLM
- ST-LLM (`liu2024spatial`)
- UrbanGPT (`li2024urbangpt`)


### 3.3 Prompting

#### 3.3.1 LLM
- LEAF (`zhao2024embracing`) - Embracing large language models in traffic flow forecasting (`zhao2024embracing`) arXiv preprint arXiv:2412.12201, 2024. [[PDF]](TBD) [[Link]](TBD)
- LLMCOD (`yu2024harnessing`) - Harnessing llms for cross-city od flow prediction (`yu2024harnessing`) SIGSPATIAL, 2024. [[PDF]](TBD) [[Link]](TBD)
- TraffiCoT-R (`alsahfi2025trafficot`) TraffiCoT-R: A framework for advanced spatio-temporal reasoning in large language models (`alsahfi2025trafficot`) Alexandria Engineering Journal, 2025. [[PDF]](TBD) [[Link]](TBD)


### 3.4 Featurization

#### 3.4.1 Diffusion Model
- DiffODGen (`rong2023complexity`) Complexity-aware large scale origin-destination network generation via diffusion model (`rong2023complexity`) arXiv preprint arXiv:2306.04873, 2023. [[PDF]](TBD) [[Link]](TBD)
- OpenDiff (`chai2024diffusion`) Diffusion model-based mobile traffic generation with open data for network planning and optimization (`chai2024diffusion`) KDD, 2024. [[PDF]](TBD) [[Link]](TBD)
- Rong et al. (`ronglarge`) - A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation (`ronglarge`) ICLR, TBD. [[PDF]](TBD) [[Link]](TBD)




## 4. Multimodal Mobility Data

![06-multimodal](assets/06-multimodal.png)

### 4.1 Vision and Trajectory
- UrbanLLaVA (`feng2025urbanllava`) UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding (`feng2025urbanllava`) arXiv preprint arXiv:2506.23219, 2025. [[PDF]](TBD) [[Link]](TBD)
- Traj-MLLM (`liu2025traj`) Traj-MLLM: Can Multimodal Large Language Models Reform Trajectory Data Mining? (`liu2025traj`) arXiv preprint arXiv:2509.00053, 2025. [[PDF]](TBD) [[Link]](TBD)
- Flame (`xu2025flame`) - Flame: Learning to navigate with multimodal llm in urban environments (`xu2025flame`) AAAI, 2025. [[PDF]](TBD) [[Link]](TBD)
- MM-RSTraj (`gao2025mm`)
- VLMLocPredictor (`zhang2025eyes`) Eyes Will Shut: A Vision-Based Next GPS Location Prediction Model by Reinforcement Learning from Visual Map Feed Back (`zhang2025eyes`) arXiv preprint arXiv:2507.18661, 2025. [[PDF]](TBD) [[Link]](TBD)
- MapGPT (`chen2024mapgpt`) Mapgpt: Map-guided prompting with adaptive path planning for vision-and-language navigation (`chen2024mapgpt`) arXiv preprint arXiv:2401.07314, 2024. [[PDF]](TBD) [[Link]](TBD)
- UGI (`xu2023urban`) Urban generative intelligence (ugi): A foundational platform for agents in embodied city environment (`xu2023urban`) arXiv preprint arXiv:2312.11813, 2023. [[PDF]](TBD) [[Link]](TBD)
- CityBench (`feng2025citybench`) - Citybench: Evaluating the capabilities of large language models for urban tasks (`feng2025citybench`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
- LLM-enhanced POI recommendation (`wang2025beyond`) Beyond Visit Trajectories: Enhancing POI Recommendation via LLM-Augmented Text and Image Representations (`wang2025beyond`) Nineteenth ACM Conference on Recommender, 2025. [[PDF]](TBD) [[Link]](TBD)

### 4.2 Text and Trajectory
- TrajSceneLLM (`ji2025trajscenellm`) TrajSceneLLM: A Multimodal Perspective on Semantic GPS Trajectory Analysis (`ji2025trajscenellm`) arXiv preprint arXiv:2506.16401, 2025. [[PDF]](TBD) [[Link]](TBD)
- Path-LLM (`wei2025path`) Path-LLM: A Multi-Modal Path Representation Learning by Aligning and Fusing with Large Language Models (`wei2025path`) ACM on Web Conference 2025, 2025. [[PDF]](TBD) [[Link]](TBD)
- Trajectory-LLM (`yang2025trajectory`) Trajectory-llm: A language-based data generator for trajectory prediction in autonomous driving (`yang2025trajectory`) ICLR, 2025. [[PDF]](TBD) [[Link]](TBD)
- TrajAgent (`du2024trajagent`) TrajAgent: An LLM-based Agent Framework for Automated Trajectory Modeling via Collaboration of Large and Small Models (`du2024trajagent`) arXiv preprint arXiv:2410.20445, 2024. [[PDF]](TBD) [[Link]](TBD)
- CoAST (`zhai2025cognitive`) Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction (`zhai2025cognitive`) arXiv preprint arXiv:2510.14702, 2025. [[PDF]](TBD) [[Link]](TBD)
- CityGPT (`feng2025citygpt`) - Citygpt: Empowering urban spatial cognition of large language models (`feng2025citygpt`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
- POI-Enhancer (`cheng2025poi`)
- D2A (`wang2024simulating`) Simulating human-like daily activities with desire-driven autonomy (`wang2024simulating`) arXiv preprint arXiv:2412.06435, 2024. [[PDF]](TBD) [[Link]](TBD)

### 4.3 Vision and Traffic
- Vision-LLM (`yang2025vision`) - Vision-LLMs for Spatiotemporal Traffic Forecasting (`yang2025vision`) arXiv preprint arXiv:2510.11282, 2025. [[PDF]](TBD) [[Link]](TBD)
- OpenDiff (`chai2024diffusion`)
- LSDM (`zhang2025lsdm`) LSDM: LLM-Enhanced Spatio-temporal Diffusion Model for Service-Level Mobile Traffic Prediction (`zhang2025lsdm`) arXiv preprint arXiv:2507.17795, 2025. [[PDF]](TBD) [[Link]](TBD)

### 4.4 Text and Traffic
- ChatTraffic (`zhang2024chattraffic`) - ChatTraffic: Text-to-traffic generation via diffusion model (`zhang2024chattraffic`) T-ITS, 2024. [[PDF]](TBD) [[Link]](TBD)
- ChatSUMO (`li2024chatsumo`) Chatsumo: Large language model for automating traffic scenario generation in simulation of urban mobility (`li2024chatsumo`) IEEE Transactions on Intelligent Vehicle, 2024. [[PDF]](TBD) [[Link]](TBD)
- UrbanMind (`liu2025urbanmind`) - UrbanMind: Urban Dynamics Prediction with Multifaceted Spatial-Temporal Large Language Models (`liu2025urbanmind`) KDD, 2025. [[PDF]](TBD) [[Link]](TBD)
- T3 (`han2024event`) - Event traffic forecasting with sparse multimodal data (`han2024event`) 32nd ACM International Conference on Mul, 2024. [[PDF]](TBD) [[Link]](TBD)
- GPT4MTS (`jia2024gpt4mts`) - Gpt4mts: Prompt-based large language model for multimodal time-series forecasting (`jia2024gpt4mts`) AAAI, 2024. [[PDF]](TBD) [[Link]](TBD)


### 4.5 Vision and Graph
- Sat2Flow (`wang2025sat2flow`) Sat2Flow: A Structure-Aware Diffusion Framework for Human Flow Generation from Satellite Imagery (`wang2025sat2flow`) arXiv preprint arXiv:2508.19499, 2025. [[PDF]](TBD) [[Link]](TBD)
- GlODGen (`rong2025satellites`) Satellites Reveal Mobility: A Commuting Origin-destination Flow Generator for Global Cities (`rong2025satellites`) arXiv preprint arXiv:2505.15870, 2025. [[PDF]](TBD) [[Link]](TBD)

### 4.6 Text and Graph
- SeMob (`chen2025semob`) - SeMob: Semantic Synthesis for Dynamic Urban Mobility Prediction (`chen2025semob`) arXiv preprint arXiv:2510.01245, 2025. [[PDF]](TBD) [[Link]](TBD)
- Ernie-GeoL (`huang2022ernie`) - Ernie-geol: A geography-and-language pre-trained model and its applications in baidu maps (`huang2022ernie`) KDD, 2022. [[PDF]](TBD) [[Link]](TBD)
- FUSE-Traffic (`yu2025fuse`) FUSE-Traffic: Fusion of Unstructured and Structured Data for Event-aware Traffic Forecasting (`yu2025fuse`) arXiv preprint arXiv:2510.16053, 2025. [[PDF]](TBD) [[Link]](TBD)
- CityFM (`balsebre2024city`) - City foundation models for learning general purpose representations from openstreetmap (`balsebre2024city`) CIKM, 2024. [[PDF]](TBD) [[Link]](TBD)



# Dataset

| Category | Dataset | Description | Geography | Statistics | Year |
|---|---|---|---|---|---|
| Discrete Sequence | [Veraset-Visits](https://www.veraset.com/datasets/visits) | POI check-in | USA | 4+ million points of interest | 2019-Present |
| Discrete Sequence | [Yelp-check-ins](https://www.yelp.com/dataset/) | POI check-in | Global | 6,990,280 Reviews | 2004-2019 |
| Discrete Sequence | Tencent~\cite{shao2024chain} | POI check-in | Beijing | 297,363,263 trajectory points | 2019 |
| Discrete Sequence | Foursquare-Global~\cite{yang2016participatory,yang2015nationtelescope} | POI check-in | Global | 33,278,683 check-ins | 2012-2013 |
| Discrete Sequence | Brightkite~\cite{cho2011friendship} | POI check-in | Global | 4,491,143 check-ins | 2008-2010 |
| Discrete Sequence | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Taxi trip record | NYC | Billions of trips | 2009-2025 |
| Discrete Sequence | [Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data) | Taxi trip record | Chicago | 212 millions trips | 2013-2023 |
| Discrete Sequence | LaDe~\cite{wu2023lade} | Delivery Records | China | 10,677K packages, 21K couriers | 2023 |
| Discrete Sequence | ChinaMobile~\cite{shao2024chain} | Cellular Trajectories | Beijing | 4,163,651 points from 1,246 users | 2017 |
| Discrete Sequence | GMove~\cite{zhang2016gmove} | Tweet check-in trajectory | New York | 0.7 million tweets | 2014 |
| Discrete Sequence | Gowalla~\cite{cho2011friendship} | human check-in trajectory | Global | 6,442,890 checkins | 2009-2010 |
| Continuous Sequence | YJMob100K-Dataset~\cite{yabe2024yjmob100k} | Human trajectory | Japan | 20k individuals, 75 days | 2023 |
| Continuous Sequence | [MTA Subway Ridership](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s) | Subway ridership | NYC | 121 millions Records | 2020-present |
| Continuous Sequence | [TaxiPorto](https://kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i) | GPS taxi trajectory | Porto | 442 taxis | 2013-2014 |
| Continuous Sequence | [DiDi-Xi'an](https://outreach.didichuxing.com) | Ride-hailing trajectories | Xi'an | 1 billion trajectories | 2016 |
| Continuous Sequence | [GAIA-Chengdu](https://outreach.didichuxing.com/research/opendata/en/) | Ride-hailing trajectories | Chengdu | 7 million ride request records | 2016 |
| Continuous Sequence | [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) | Human GPS trajectory | China | 17,621 trajectories | 2007-2012 |
| Continuous Sequence | T-Drive~\cite{yuan2011driving} | Taxi GPS trajectory | Beijing | 169,984 trajectories, 10,357 taxis | 2008 |
| Continuous Sequence | TaxiBJ~\cite{zhang2017deep} | Taxi flow | Beijing | 22,459 time intervals | 2013-2016 |
| Continuous Sequence | [SafeGraph](https://www.safegraph.com) | Visits per month by POI | USA | 84 million | 2019-Present |
| Continuous Sequence | [Advan Monthly Patterns](https://docs.deweydata.io/docs/advan-research-monthly-patterns) | Visits per month by POI | USA | 1B | 2019-Present |
| Continuous Sequence | [Advan Weekly Patterns](https://docs.deweydata.io/docs/advan-research-weekly-patterns) | Visits per week by POI | USA | 2.3B | 2018-Present |
| Spatio-Temporal Graph | [PEMS](https://pems.dot.ca.gov/) | Mobility network | California | 2 GB per day | 1998-present |
| Spatio-Temporal Graph | [NYC Yellow Taxi](https://doi.org/10.5281/zenodo.17089134) | Trip-Hourly-Count flow | NYC | 36.4 million total trip volumes | 2011-2024 |
| Spatio-Temporal Graph | [CHI-Taxi](https://data.cityofchicago.org/Transportation/Taxi-Trips-2024/sa9s-wkhk) | Taxi demand | Chicago | 77 nodes | 2024 |
| Spatio-Temporal Graph | LargeST~\cite{liu2023largest} | Vehicle traffic flow | California | 525,888 time frames | 2017-2021 |
| Spatio-Temporal Graph | COVID-19 Human Mobility~\cite{kang2020multiscale} | OD flow network | USA | 3 geographic scales | 2020 |
| Spatio-Temporal Graph | LODES-7.5~\cite{LODES7.5_2021} | Commuting OD flow network | USA | 12 OD files for a state-year | 2002-2019 |
| Spatio-Temporal Graph | [BikeNYC](https://citibikenyc.com/system-data) | Bike flow | NYC | Millions Monthly | 2014 |
| Spatio-Temporal Graph | BJER4~\cite{yu2017spatio} | Road network traffic speed dataset | Beijing | 12 roads | 2014 |
| Spatio-Temporal Graph | METR-LA~\cite{li2017diffusion} | Traffic speed network dataset | Los Angeles | 6,519,002 points | 2012 |
| Spatio-Temporal Graph | UK mobility flow~\cite{simini2021deep} | Commuting OD flow network | UK | 30,008,634 commuters | 2011 |
| Spatio-Temporal Graph | Italy mobility flow~\cite{simini2021deep} | Commuting OD flow network | Italy | 15,003,287 commuters | 2011 |
| Multimodal Mobility | [Yelp Dataset](https://www.yelp.com/dataset/) | Check-in + Text Reviews | Global | 6,990,280 Reviews | 2004-2019 |
| Multimodal Mobility | nuScenes~\cite{caesar2020nuscenes} | Vision + Trajectories | Boston, Singapore | 1000 scenes, 1.4M images | 2019 |
| Multimodal Mobility | [Waymo Open Motion Dataset](https://waymo.com/open/download) | Vision + Trajectories | Six U.S. cities | 570+ hours at 10 Hz | 2020-2023 |
| Multimodal Mobility | TartanAviation~\cite{patrikar2025image} | Images + Trajectories | Pittsburgh | 3.1M images, 661 days of trajectory | 2020-2023 |
| Multimodal Mobility | GlODGen~\cite{rong2025satellites} | Vision + OD Flow | Global | synthetic data | 2025 |
| Multimodal Mobility | Earth AI~\cite{bell2025earth} | Vision + population + environment | Global | 10-meter resolution | 2025 |
| Multimodal Mobility | NetMob25~\cite{chasse2025netmob25} | Population + trip description + trajectories | Greater Paris area | 500 million high frequency points | 2022-2023 |
| Multimodal Mobility | BostonWalks~\cite{meister2025bostonwalks} | Population + trip + activity | Boston metropolitan area | 155,000 trips, 990 participants | 2023 |
| Multimodal Mobility | Breadcrumbs~\cite{moro2019breadcrumbs} | Population + trajectory + POI labeling | Lausanne | 46,380,042 records, 81 users | 2018 |
| Multimodal Mobility | RECORD MultiSensor Study~\cite{chaix2019combining} | Population + trajectory + semantic trip annotations | Paris region | 21,163 segments of observation | 2013-2015 |





# References from Other AI Communities


# Citation

