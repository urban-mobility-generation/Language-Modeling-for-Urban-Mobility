# Language-Modeling-for-Urban-Mobility-A-Data-Centric-Survey
> Language Modeling for Urban Mobility: A Data-Centric Review and Guidelines


> This repository contains a collection of resources and papers on applying language modeling paradigms to urban mobility scenarios.


## Contents

- [1. Discrete Mobility Sequence](#1-discrete-mobility-sequence)
  - [1.1 Tokenization](#11-tokenization)
    - [1.1.1 Early LM](#111-early-lm)
    - [1.1.2 Pretrained LM](#112-pretrained-lm)
    - [1.1.3 LLM](#113-llm)
  - [1.2 Encoding](#12-encoding)
    - [1.2.1 Early-stage LM](#121-early-stage-lm)
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
- [3. Spatial-temporal Graph](#3-spatial-temporal-graph)
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

## 1. Discrete Mobility Sequence

### 1.1 Tokenization

#### 1.1.1 Early LM
- T-gram (`hsieh2015t`)
- DNTM (`farrahi2012extracting`)

#### 1.1.2 Pretrained LM

- **Attention**
  - MoveSim (`feng2020learning`)
  - TPG (`luo2023timestamps`)

- **Masked LM**
  - CTLE (`lin2021pre`)
  - Wepos (`guo2022wepos`)
  - Yang et al. (`yang2024applying`)
  - GREEN (`zhou2025grid`)
  - TrajBERT (`trajbert2023`)
  - Yang et al. (`yang2024applying`)

- **Transformer Decoder**
  - MobilityGPT (`mobilitygpt2024`)
  - GeoFormer (`solatorio2023geoformer`)
  - LMTAD (`mbuya2024trajectory`)
  - Kobayashi et al. (`kobayashi2023modeling`)
  - Traj-LLM (`lan2024traj`)
  - TrajLearn (`nadiri2025trajlearn`)

#### 1.1.3 LLM
- GNPR-SID (`wang2025generativekdd25`)
- RHYTHM (`he2025rhythm`)
- MobGLM (`zhang2024mobglm`)
- MobilityGPT (`mobilitygpt2024`)
- Geo-Llama (`li2024geo`)


### 1.2 Encoding

#### 1.2.1 Early-stage LM
- Deepmove (`feng2018deepmove`)
- STRNN (`liu2016predicting`)
- LSTPM (`sun2020go`)
- STAN (`Luo2021stan`)

#### 1.2.2 Pretrained LM

- **Masked LM**
  - LP-BERT (`suzuki2024cross`)
  - TraceBERT (`crivellari2022tracebert`)
  - CTLE (`lin2021pre`)
  - GREEN (`zhou2025grid`)

- **Causal Attention**
  - MobTCast (`xue2021mobtcast`)
  - MoveSim (`feng2020learning`)
  - AttnMove (`xia2021attnmove`)

#### 1.2.3 LLM
- LLMEmb (`liu2025llmemb`)
- Mobility-LLM (`mobilityllm2024`)
- NextLocLLM (`nextlocllm2025`)
- GSTM-HMU (`luo2025gstm`)

#### 1.2.4 Diffusion Model
- Cardiff (`guo2025leveraging`)
- Diff-POI (`qin2023diffusion`)
- AutoSTDiff (`xu2025autostdiff`)
- DiffMove (`long2025diffmove`)
- GenMove (`long2025one`)
- Diff-DGMN (`zuo2024diff`)
- DCPR (`long2024diffusion`)
- Traveller (`luo2025traveller`)
- TrajGDM (`chu2024simulating`)


### 1.3 Prompting

#### 1.3.1 LLM

- **As Representor**
  - Poi-enhancer (`cheng2025poi`)
  - LLM-Mob (`wang2023would`)
  - TrajCogn (`zhou2024trajcogn`)

- **As Predictor**
  - TPP-LLM (`liu2024tpp`)
  - CoMaPOI (`zhong2025comapoi`)
  - AgentMove (`feng2024agentmove`)
  - Feng et al. (`feng2024move`)
  - LLM4Poi (`li2024large`)
  - CSA-Rec (`wang2025collaborative`)
  - LAMP (`balsebre2024lamp`)
  - Zhang et al. (`zhang2023large`)
  - Mo et al. (`mo2023large`)
  - POI GPT (`kim2024poi`)
  - Chen et al. (`chen2025toward`)
  - DelayPTC-LLM (`chen2024delayptc`)

- **Generator**
  - Liu et al. (`liu2025aligning`)
  - CoPB (`shao2024chain`)
  - Liu et al. (`liu2023can`)
  - Bhandari et al. (`bhandari2024urban`)
  - Zheng et al. (`zheng2025urban`)

- **LLM Agents**
  - LLM-HABG (`meng2025behavior`)
  - PathGPT (`marcelyn2025pathgpt`)
  - LLMTraveler (`wang2024ai`)
  - GATSim (`liu2025gatsim`)
  - MobAgent (`li2024more`)
  - CitySim (`bougie2025citysim`)
  - TravelPlanner (`xie2024travelplanner`)
  - IDM-GPT (`yang2025independent`)



## 2. Continuous Mobility Sequence

### 2.1 Discrete Tokenization (Quantization)

#### 2.1.1 Pretrained LM

- **Encoder-based (BERT-like)**
  - Giuliari (`giuliari2021transformer`)
  - BERT4Traj (`yang2025bert4traj`)

- **Decoder-based (GPT-like)**
  - MotionLM (`seff2023motionlm`)
  - RAW (`zhang2023regions`)

- **Encoder–Decoder-based**
  - UniTraj (`zhu2024unitraj`)

#### 2.1.2 LLM
- LMTraj (`bae2024can`)
- RouteLLM (`hallgarten2025routellm`)
- QT-Mob (`chen2025enhancing`)
- CAMS (`du2025cams`)
- AutoTimes (`liu2024autotimes`)  <!-- traffic time series -->


### 2.2 Encoding

#### 2.2.1 Pretrained LM
- BERT4Traj (`yang2025bert4traj`)
- EETG-SVAE (`zhang2025end`)
- LM-TAD (`mbuya2024trajectory`)
- Musleh et al. (`musleh2022towards`)
- UrbanGPT (`li2024urbangpt`)
- UniST (`yuan2024unist`)
- FlashST (`li2024flashst`)
- Traffic-Twitter Transformer (`tsai2022traffic`)
- FlowDistill (`yu2025flowdistill`)
- Cao et al. (`cao2021bert`)
- Ma et al. (`ma2025urban`)
- TrafficBERT (`jin2021trafficbert`)
- ST-LLM+ (`liu2025st`)
- MDTI (`liu2025multimodal`)

#### 2.2.2 LLM
- TPLLM (`ren2024tpllm`)
- LLM-TFP (`cheng2025llm`)
- NextLocLLM (`nextlocllm2025`)
- Liao et al. (`liao2025next`)


### 2.3 Prompting

#### 2.3.1 LLM
- Zhang et al. (`zhang2024large`)
- GPT-J (`ji2024evaluating`)
- GeoLLM (`manvi2023geollm`)
- AuxMobLCast (`xue2022leveraging`)
- Wang et al. (`wang2025event`)

- **Prediction**
  - LLM-MPE (`liang2024exploring`)
  - STCInterLLM (`li2025causal`)
  - xTP-LLM (`guo2024towards`)
  - Cai et al. (`cai2024temporal`)
  - LLM4PT (`wu2025llm4pt`)
  - TransLLM (`leng2025transllm`)

- **Generation**
  - LLMob (`jiawei2024large`)


### 2.4 Featurization

#### 2.4.1 Diffusion Model
- CoDiffMob (`codiffmob2025`)
- ControlTraj (`zhu2024controltraj`)
- DiffTraj (`difftraj2023`)
- Cardiff (`guo2025leveraging`)
- UniMob (`long2025universal`)



## 3. Spatial-temporal Graph

### 3.1 Tokenization

#### 3.1.1 Pretrained LM
- UniFlow (`yuan2024uniflow`)
- RePST (`wang2024repst`)
- CompactST (`han2025scalable`)
- STD-PLM (`huang2025std`)

#### 3.1.2 LLM
- STG-LLM (`liu2024can`)
- ST-LLM (`liu2024spatial`)


### 3.2 Encoding

#### 3.2.1 Pretrained LM
- STGormer (`zhou2024navigating`)
- STGLLM-E (`rong2024edge`)
- CityCAN (`wang2024citycan`)
- STTNs (`xu2020spatial`)
- ST-LINK (`jeon2025st`)

#### 3.2.2 LLM
- ST-LLM (`liu2024spatial`)
- UrbanGPT (`li2024urbangpt`)


### 3.3 Prompting

#### 3.3.1 LLM
- LEAF (`zhao2024embracing`)
- LLMCOD (`yu2024harnessing`)
- TraffiCoT-R (`alsahfi2025trafficot`)


### 3.4 Featurization

#### 3.4.1 Diffusion Model
- DiffODGen (`rong2023complexity`)
- OpenDiff (`chai2024diffusion`)
- Rong et al. (`ronglarge`)



## 4. Multimodal Mobility Data

### 4.1 Vision and Trajectory
- UrbanLLaVA (`feng2025urbanllava`)
- Traj-MLLM (`liu2025traj`)
- Flame (`xu2025flame`)
- MM-RSTraj (`gao2025mm`)
- VLMLocPredictor (`zhang2025eyes`)
- MapGPT (`chen2024mapgpt`)
- UGI (`xu2023urban`)
- CityBench (`feng2025citybench`)
- LLM-enhanced POI recommendation (`wang2025beyond`)

### 4.2 Text and Trajectory
- TrajSceneLLM (`ji2025trajscenellm`)
- Path-LLM (`wei2025path`)
- Trajectory-LLM (`yang2025trajectory`)
- TrajAgent (`du2024trajagent`)
- CoAST (`zhai2025cognitive`)
- CityGPT (`feng2025citygpt`)
- POI-Enhancer (`cheng2025poi`)
- D2A (`wang2024simulating`)

### 4.3 Vision and Traffic
- Vision-LLM (`yang2025vision`)
- OpenDiff (`chai2024diffusion`)
- LSDM (`zhang2025lsdm`)

### 4.4 Text and Traffic
- ChatTraffic (`zhang2024chattraffic`)
- ChatSUMO (`li2024chatsumo`)
- UrbanMind (`liu2025urbanmind`)
- T3 (`han2024event`)
- GPT4MTS (`jia2024gpt4mts`)

### 4.5 Vision and Graph
- Sat2Flow (`wang2025sat2flow`)
- GlODGen (`rong2025satellites`)

### 4.6 Text and Graph
- SeMob (`chen2025semob`)
- Ernie-GeoL (`huang2022ernie`)
- FUSE-Traffic (`yu2025fuse`)
- CityFM (`balsebre2024city`)

