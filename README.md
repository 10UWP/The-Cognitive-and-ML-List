# :chart: The-Cognitive-and-ML-List

Cognitive Services, ML, Machine Learning, Data Ingestion, Azure, Docker, Container, R, Python, C#, Java, Hadoop, Spark etc

# 1 - IOT 

What will become the primary source of Data for ML by orders of magnitude in the next decades



# 1 - Signals - Vision, Audio etc


# 1 - DoD - Data on Disk


# 2 - Data Ingestion

[River View](https://github.com/htm-community/river-view) Public Temporal Streaming Data Service Framework http://data.numenta.org/ -  River View is a Public Temporal Streaming Data Service Framework (yes, that's a mouthful!). It provides a pluggable interface for users to expose temporal data streams in a time-boxed format that is easily query-able. It was built to provide a longer-lasting historical window for public data sources that provide only real-time data snapshots, especially for sensor data from public government services like weather, traffic, and geological data. River View fetches data from user-defined Rivers at regular intervals, populating a local Redis database. This data is provided in a windowed format, so that data older than a certain configured age is lost. But the window should be large enough to provide enough historical data to potentially train machine intelligence models on the data patterns within it.


[Gobblin](https://github.com/linkedin/gobblin) Universal data ingestion framework for Hadoop. https://github.com/linkedin/gobblin/wiki - Gobblin is a universal data ingestion framework for extracting, transforming, and loading large volume of data from a variety of data sources, e.g., databases, rest APIs, FTP/SFTP servers, filers, etc., onto Hadoop. Gobblin handles the common routine tasks required for all data ingestion ETLs, including job/task scheduling, task partitioning, error handling, state management, data quality checking, data publishing, etc. Gobblin ingests data from different data sources in the same execution framework, and manages metadata of different sources all in one place. This, combined with other features such as auto scalability, fault tolerance, data quality assurance, extensibility, and the ability of handling data model evolution, makes Gobblin an easy-to-use, self-serving, and efficient data ingestion framework.

[StreamSets DataCollector](https://github.com/streamsets/datacollector) StreamSets DataCollector - Continuous big data ingest infrastructure http://www.streamsets.com - StreamSets Data Collector is an enterprise grade, open source, continuous big data ingestion infrastructure. It has an advanced and easy to use User Interface that lets data scientists, developers and data infrastructure teams easily create data pipelines in a fraction of the time typically required to create complex ingest scenarios. Out of the box, StreamSets Data Collector reads from and writes to a large number of end-points, including S3, JDBC, Hadoop, Kafka, Cassandra and many others. You can use Python, Javascript and Java Expression Language in addition to a large number of pre-built stages to transform and process the data on the fly. For fault tolerance and scale out, you can setup data pipelines in cluster mode and perform fine grained monitoring at every stage of the pipeline.



# 3 - Processing

The actual ML, Cognitive, AI part

[caffe](https://github.com/BVLC/caffe) Caffe: a fast open framework for deep learning. http://caffe.berkeleyvision.org/  - Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and community contributors.

[Documentation for Microsoft Cognitive Services](https://github.com/Microsoft/Cognitive-Documentation) This repo contains the documentaiton for Microsoft Cognitive Services, which includes the former Project Oxford APIs. You can also check out our SDKs & Samples on our website.  Don't see what you're looking for? We're working on expanding our offerings and would love to hear from you what APIs, docs, SDKs, and samples you want to see next. Let us know on the Cognitive Services UserVoice Forum.

[Intelligent Kiosk Sample](https://github.com/Microsoft/Cognitive-Samples-IntelligentKiosk) Welcome to the Intelligent Kiosk Sample! Here you will find several demos showcasing workflows and experiences built on top of the Microsoft Cognitive Services. https://www.microsoft.com/cognitive-services The Intelligent Kiosk Sample is a collection of demos showcasing workflows and experiences built on top of the Microsoft Cognitive Services. Most of the experiences are hands-free and autonomous, using the human faces in front of a web camera as the main form of input (thus the word "kiosk" in the name).

[Computational Network Toolkit (CNTK)](https://github.com/Microsoft/CNTK) CNTK (http://www.cntk.ai/), the Computational Network Toolkit by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

[Multiverso](https://github.com/Microsoft/multiverso) Multiverso is a parameter server based framework for training machine learning models on big data with numbers of machines. It is currently a standard C++ library and provides a series of friendly programming interfaces. With such easy-to-use APIs, machine learning researchers and practitioners do not need to worry about the system routine issues such as distributed model storage and operation, inter-process and inter-thread communication, multi-threading management, and so on. Instead, they are able to focus on the core machine learning logics: data, model, and training. For more details, please view our website http://www.dmtk.io.

[Distributed Machine Learning Toolkit](https://github.com/Microsoft/DMTK) DMTK includes the following projects:
- [DMTK framework(Multiverso)](https://github.com/Microsoft/multiverso): The parameter server framework for distributed machine learning.
- [LightLDA](https://github.com/Microsoft/lightlda): Scalable, fast and lightweight system for large-scale topic modeling.
- [Distributed word embedding](https://github.com/Microsoft/distributed_word_embedding): Distributed algorithm for word embedding.
- [Distributed skipgram mixture](https://github.com/Microsoft/distributed_skipgram_mixture): Distributed algorithm for multi-sense word embedding.




[Project Malmo](https://github.com/Microsoft/malmo) Project Malmo is a platform for Artificial Intelligence experimentation and research built on top of Minecraft. We aim to inspire a new generation of research into challenging new problems presented by this unique environment. --- For installation instructions, scroll down to *Getting Started* below, or visit the project page for more information: https://www.microsoft.com/en-us/research/project/project-malmo/

[aerosolve](https://github.com/airbnb/aerosolve) A machine learning package built for humans. http://airbnb.github.io/aerosolve/ - A machine learning library designed from the ground up to be human friendly. It is different from other machine learning libraries in the following ways:
-    A thrift based feature representation that enables pairwise ranking loss and single context multiple item representation.
-    A feature transform language gives the user a lot of control over the features
-    Human friendly debuggable models
-    Separate lightweight Java inference code
-    Scala code for training
-    Simple image content analysis code suitable for ordering or ranking images
-    This library is meant to be used with sparse, interpretable features such as those that commonly occur in search (search keywords, filters) or pricing (number of rooms, location, price). It is not as interpretable with problems with very dense non-human interpretable features such as raw pixels or audio samples.

[Deep Scalable Sparse Tensor Network Engine (DSSTNE)](https://github.com/amznlabs/amazon-dsstne) Deep Scalable Sparse Tensor Network Engine (DSSTNE) is an Amazon developed library for building Deep Learning (DL) machine learning (ML) models - DSSTNE (pronounced "Destiny") is an open source software library for training and deploying deep neural networks using GPUs. Amazon engineers built DSSTNE to solve deep learning problems at Amazon's scale. DSSTNE is built for production deployment of real-world deep learning applications, emphasizing speed and scale over experimental flexibility.


[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning. http://hunch.net/~vw/

[TensorFlow](https://github.com/tensorflow/tensorflow) Computation using data flow graphs for scalable machine learning http://tensorflow.org - TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.  TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

[scikit-learn](https://github.com/scikit-learn/scikit-learn) scikit-learn is a Python module for machine learning built on top of SciPy 
[Pattern](https://github.com/clips/pattern) Web mining module for Python, with tools for scraping, natural language processing, machine learning, network analysis and visualization. http://www.clips.ua.ac.be/pages/pattern 
-    Data Mining: web services (Google, Twitter, Wikipedia), web crawler, HTML DOM parser
-    Natural Language Processing: part-of-speech taggers, n-gram search, sentiment analysis, WordNet
-    Machine Learning: vector space model, clustering, classification (KNN, SVM, Perceptron)
-    Network Analysis: graph centrality and visualization.

[Numenta Platform for Intelligent Computing](https://github.com/numenta/nupic) Numenta Platform for Intelligent Computing: a brain-inspired machine intelligence platform, and biologically accurate neural network based on cortical learning algorithms. http://numenta.org/ - The Numenta Platform for Intelligent Computing (NuPIC) is a machine intelligence platform that implements the HTM learning algorithms. HTM is a detailed computational theory of the neocortex. At the core of HTM are time-based continuous 
learning algorithms that store and recall spatial and temporal patterns. NuPIC is suited to a variety of problems, particularly anomaly detection and prediction of streaming data sources.
- [Metis](https://github.com/mrs110/Metis) Speech Recognition Project Using Numenta's NuPIC
- [Cell Vis](https://github.com/numenta/cell-viz) This is an attempt at recreating an old illustration. See it live at https://numenta.github.io/cell-viz/ or fiddle with it at http://jsfiddle.net/7tbm3mv1/4/.
- [HTM School Visualizations](https://github.com/htm-community/htm-school-viz)  Visualizations of Sparse Distributed Representations
- [Numenta Apps](https://github.com/numenta/numenta-apps) HTM based applications and support libraries. Includes Grok for IT Analytics and Grok for Stocks (code name "NabVizTaurus"). http://numenta.com
- [NabViz](https://github.com/y-takashina/NabViz) A visualization tool for Numenta Anomaly Benchmark (NAB). http://numenta.org/nab/
- [Flight Path Anomaly Detection](https://github.com/shuai-zh/flight-path-anomaly-detection-system) A flight path anomaly detection simulation application using nupic(Numenta Platform for Intelligent Computing) and Cesium.js
- [The Numenta Anomaly Benchmark](https://github.com/numenta/NAB) contains the data and scripts comprising the Numenta Anomaly Benchmark (NAB). NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications. It is comprised of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications. Included are the tools to allow you to easily run NAB on your own anomaly detection algorithms;
- [https://github.com/binarybarry/HTM-Camera-Toolkit](https://github.com/binarybarry/HTM-Camera-Toolkit) The HTM Camera Toolkit is a research application that allows easy experimentation of Numenta's Hierarchical Temporal Memory (HTM) algorithms using real world video input from a camera/webcam.
- [nupic.vision](https://github.com/numenta/nupic.vision) Tools for using Numenta/nupic on visual problems like image recognition.
- [AI-909](https://github.com/TaylorPeer/AI-909) Intelligent drum machine powered by Numenta's Hierarchical Temporal Memory  
- [Hypersearch](https://github.com/numenta/hypersearch) A particle swarm optimization library created by Numenta for hyperparameter optimization.

### Javascript

- [HTM](https://github.com/sebjwallace/HTM) progressive implimentation of Numenta's HTM model. Visualizations are included for the spatial pooler (then later temporal pooler).

### C++/C
- [LibHtm](https://github.com/pdJeeves/LibHtm) An implementation of Numenta's Hierarchical Temporal Memory in C
- [HTMCLA](https://github.com/MichaelFerrier/HTMCLA) A C++ implementation of Numenta's Hierarchical Temporal Memory (HTM) Cortical Learning Algorithm (CLA). Uses Qt for user interface.

### Java
- [htm.java](https://github.com/numenta/htm.java) Hierarchical Temporal Memory implementation in Java - an official Community-Driven Java port of the Numenta Platform for Intelligent Computing (NuPIC).
- [HTM](https://github.com/zygon4/htm) An attempt at implementing the Numenta HTM model in java

### C#
- [HTM.net](https://github.com/Zuntara/HTM.Net) Ported version of Numenta's Hierachical Temporal Memory Engine
- [NabViz](https://github.com/y-takashina/NabViz) A visualization tool for Numenta Anomaly Benchmark (NAB). http://numenta.org/nab/
- [dooHTM](https://github.com/avogab/dooHTM) The dooHTM is a c# Hello World application that permits a first glance at Numenta's Hierarchical Temporal Memory (HTM) algorithms using very simple generated motion images.




[TPOT](https://github.com/rhiever/tpot) A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming. http://rhiever.github.io/tpot/ - TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data. Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there. TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

[Accord.NET Framework](https://github.com/accord-net/framework) The Accord.NET Framework http://accord-framework.net - The Accord.NET Framework provides machine learning, mathematics, statistics, computer vision, computer audition, and several scientific computing related methods and techniques to .NET. The project extends the popular AForge.NET Framework providing a more complete scientific computing environment.

[Shogun Machine Learning Toolbox](https://github.com/shogun-toolbox/shogun) The Shogun Machine Learning Toolbox (Source Code) http://www.shogun-toolbox.org
- The Shogun Machine learning toolbox provides a wide range of unified and efficient Machine Learning (ML) methods. The toolbox seamlessly allows to easily combine multiple data representations, algorithm classes, and general purpose tools. This enables both rapid prototyping of data pipelines and extensibility in terms of new algorithms. We combine modern software architecture in C++ with both efficient low-level computing backends and cutting edge algorithm implementations to solve large-scale Machine Learning problems (yet) on single machines.
- One of Shogun's most exciting features is that you can use the toolbox through a unified interface from C++, Python, Octave, R, Java, Lua, C#, etc. This not just means that we are independent of trends in computing languages, but it also lets you use Shogun as a vehicle to expose your algorithm to multiple communities. We use SWIG to enable bidirectional communication between C++ and target languages. Shogun runs under Linux/Unix, MacOS, Windows.
- Originally focussing on large-scale kernel methods and bioinformatics (for a list of scientific papers mentioning Shogun, see here), the toolbox saw massive extensions to other fields in recent years. It now offers features that span the whole space of Machine Learning methods, including many classical methods in classification, regression, dimensionality reduction, clustering, but also more advanced algorithm classes such as metric, multi-task, structured output, and online learning, as well as feature hashing, ensemble methods, and optimization, just to name a few. Shogun in addition contains a number of exclusive state-of-the art algorithms such as a wealth of efficient SVM implementations, Multiple Kernel Learning, kernel hypothesis testing, Krylov methods, etc. All algorithms are supported by a collection of general purpose methods for evaluation, parameter tuning, preprocessing, serialisation & I/O, etc; the resulting combinatorial possibilities are huge. See our feature list for more details.
- The wealth of ML open-source software allows us to offer bindings to other sophisticated libraries including: LibSVM/LibLinear, SVMLight, LibOCAS, libqp, VowpalWabbit, Tapkee, SLEP, GPML and more. See our list of [integrated external libraries](http://www.shogun-toolbox.org/page/about/contributions)

[mlpack](https://github.com/mlpack/mlpack) mlpack: a scalable C++ machine learning library - mlpack is an intuitive, fast, scalable C++ machine learning library, meant to be a machine learning analog to LAPACK. It aims to implement a wide array of machine learning methods and functions as a "swiss army knife" for machine learning researchers. The mlpack website can be found at http://www.mlpack.org and contains numerous tutorials and extensive documentation. 

[Deep Neural Networks as a Service](https://github.com/claritylab/djinn) - Deep Neural Networks (DNNs) as a Service http://www.djinn.clarity-lab.org - DjiNN and Tonic is a Deep Neural Network (DNN) based web-service to execute DNN inference. [Tonic Suite](http://djinn.clarity-lab.org/tonic-suite/) is a suite of 7 applications that use the service. Tonic Suite includes image, speech, and natural language processing applications that use a common DNN backend as their machine learning component. DjiNN and Tonic is developed by Clarity Lab at the University of Michigan.

[ConvNetJS](https://github.com/karpathy/convnetjs) - Deep Learning in Javascript. Train Convolutional Neural Networks (or ordinary ones) in your browser. [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/) is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:
-    Common Neural Network modules (fully connected layers, non-linearities)
-    Classification (SVM/Softmax) and Regression (L2) cost functions
-    Ability to specify and train Convolutional Networks that process images
-    An experimental Reinforcement Learning module, based on Deep Q Learning

[ConvNetSharp](https://github.com/cbovar/ConvNetSharp) C# port of ConvNetJS. You can use ConvNetSharp to train and evaluate convolutional neural networks (CNN).

[mxnet](https://github.com/dmlc/mxnet) Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Scala, Go, Javascript and more http://mxnet.io - MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to mix the flavours of symbolic programming and imperative programming to maximize efficiency and productivity. In its core, a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. The library is portable and lightweight, and it scales to multiple GPUs and multiple machines. MXNet is also more than a deep learning project. It is also a collection of blue prints and guidelines for building deep learning system, and interesting insights of DL systems for hackers.

[Neon](https://github.com/NervanaSystems/neon) Fast, scalable, easy-to-use Python based Deep Learning Framework by Nervana™ http://neon.nervanasys.com/ - neon is Nervana's Python based Deep Learning framework and achieves the fastest performance on modern deep neural networks such as AlexNet, VGG and GoogLeNet. Designed for ease-of-use and extensibility.
-    Tutorials and iPython notebooks to get users started with using neon for deep learning.
-    Support for commonly used layers: convolution, RNN, LSTM, GRU, BatchNorm, and more.
-    Model Zoo contains pre-trained weights and example scripts for start-of-the-art models, including: VGG, Reinforcement learning, Deep Residual Networks, Image Captioning, Sentiment analysis, and more.
-    Swappable hardware backends: write code once and then deploy on CPUs, GPUs, or Nervana hardware
- [neon course](https://github.com/NervanaSystems/neon_course) contains several jupyter notebooks to help users learn to use neon, our deep learning framework. For more information, see our documentation and our API.
- [aeon](https://github.com/NervanaSystems/aeon) module for data loading and transforming
- [ModelZoo](https://github.com/NervanaSystems/ModelZoo) contains a number of standard deep learning models that can be run with the neon libraries.

[Lasagne](https://github.com/Lasagne/Lasagne) Lightweight library to build and train neural networks in [Theano](http://www.deeplearning.net/software/theano/) http://lasagne.readthedocs.org/  Lasagne is a lightweight library to build and train neural networks in Theano. Its main features are:
-    Supports feed-forward networks such as Convolutional Neural Networks (CNNs), recurrent networks including Long Short-Term Memory (LSTM), and any combination thereof
-    Allows architectures of multiple inputs and multiple outputs, including auxiliary classifiers
-    Many optimization methods including Nesterov momentum, RMSprop and ADAM
-    Freely definable cost function and no need to derive gradients due to Theano's symbolic differentiation
-    Transparent support of CPUs and GPUs due to Theano's expression compiler

[Theano](https://github.com/Theano/Theano) Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It can use GPUs and perform efficient symbolic differentiation. http://www.deeplearning.net/software/theano

[Pipeline.IO](https://github.com/fluxcapacitor/pipeline) Real-time, End-to-End, Advanced Analytics and Machine Learning Recommendation Pipeline http://pipeline.io

[Apache Hadoop](https://github.com/apache/hadoop) For the latest information about Hadoop, please visit our website at: http://hadoop.apache.org/core/ and our wiki, at:  http://wiki.apache.org/hadoop/

[Apache Spark](https://github.com/apache/spark) Spark is a fast and general cluster computing system for Big Data. It provides high-level APIs in Scala, Java, Python, and R, and an optimized engine that supports general computation graphs for data analysis. It also supports a rich set of higher-level tools including Spark SQL for SQL and DataFrames, MLlib for machine learning, GraphX for graph processing, and Spark Streaming for stream processing. http://spark.apache.org/

[Multiworld Testing Decision Service](https://github.com/Microsoft/mwt-ds) https://github.com/Microsoft/mwt-ds/wiki [MWT White Paper: Methodology; system design; how to make the DS fit your application; past deployments; experimental evaluation](https://github.com/Microsoft/mwt-ds/raw/master/images/MWT-WhitePaper.pdf)

    How to tailor the DS to YOUR application (conceptually, in terms of machine learning)
    How to track competing predictions in real time.
    How to extract your model for experimentation outside of the DS
    How to delete your instance to save money, conserve your quota, etc.

# 4 - Bots

[superscript](https://github.com/superscriptjs/superscript) A dialogue engine for creating chat bots http://superscriptjs.com -  SuperScript is a dialog system + bot engine for creating human-like conversation chat bots. It exposes an expressive script for crafting dialogue and features text-expansion using wordnet and Information Retrieval and extraction using ConceptNet.

[SimpleDS](https://github.com/cuayahuitl/SimpleDS) A Simple Deep Reinforcement Learning Dialogue System - SimpleDS is a simple dialogue system trained with deep reinforcement learning. In contrast to other dialogue systems, this system selects dialogue actions directly from raw (noisy) text of the last system and user responses. The motivation is to train dialogue agents with as little human intervention as possible. 

[Bot Builder SDK](https://github.com/Microsoft/BotBuilder) The Microsoft Bot Builder SDK is one of three main components of the Microsoft Bot Framework. The Microsoft Bot Framework provides just what you need to build and connect intelligent bots that interact naturally wherever your users are talking, from text/SMS to Skype, Slack, Office 365 mail and other popular services. http://botframework.com - Bots (or conversation agents) are rapidly becoming an integral part of one’s digital experience – they are as vital a way for users to interact with a service or application as is a web site or a mobile experience. Developers writing bots all face the same problems: bots require basic I/O; they must have language and dialog skills; and they must connect to users – preferably in any conversation experience and language the user chooses. The Bot Framework provides tools to easily solve these problems and more for developers e.g., automatic translation to more than 30 languages, user and conversation state management, debugging tools, an embeddable web chat control and a way for users to discover, try, and add bots to the conversation experiences they love.

[AzureBot](https://github.com/Microsoft/AzureBot) This is the source code which runs the Microsoft AzureBot. The AzureBot isn't public yet, but stay tuned. http://aka.ms/AzureBot - The AzureBot was created to improve the productivity of any developer, admin, or team working with Azure. It is not currently publicly available, but you can follow our Developer Set Up to run it yourself and contribute. This first implementation focuses on authenticating to the user's Azure subscription, selecting and switching subscriptions, starting and stopping RM-based virtual machines, and listing and starting Azure Automation runbooks. 
    
# 4 - Visualization - BI, Dashboards, Interactive, 3D Augmented, 3D VR

[HoloToolkit](https://github.com/Microsoft/HoloToolkit) The HoloToolkit is a collection of scripts and components intended to accelerate the development of holographic applications targeting Windows Holographic.

[Holographic Academy](https://github.com/Microsoft/HolographicAcademy) This will be the home of all code assets necessary for the Holographic Academy. All of the courses can be found in their own branches. This is so developers can download zip folders for these tutorials from the Academy documentation.

[Seriously.js](https://github.com/brianchirls/Seriously.js) A real-time, node-based video effects compositor for the web built with HTML5, Javascript and WebGL http://seriouslyjs.org - Seriously.js is a real-time, node-based video compositor for the web. Inspired by professional software such as After Effects and Nuke, Seriously.js renders high-quality video effects, but allows them to be dynamic and interactive.

[Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) This is the code required to run the Deep Visualization Toolbox, as well as to generate the neuron-by-neuron visualizations using regularized optimization.

# Supporting Infrastructure and Glue Bits


## API

[Mobius](https://github.com/Microsoft/Mobius) C# language binding and extensions to Apache Spark. Mobius provides C# language binding to Apache Spark enabling the implementation of Spark driver program and data processing operations in the languages supported in the .NET framework like C# or F#.

[Language Understanding Intelligent Service API](https://github.com/Microsoft/Cognitive-LUIS-Windows) Windows (.Net) SDK for the Microsoft Language Understanding Intelligent Service API, part of Congitive Services http://www.microsoft.com/cognitive-services/en-us/language-understanding-intelligent-service-luis - LUIS is a service for language understanding that provides intent classification and entity extraction. In order to use the SDK you first need to create and publish an app on www.luis.ai where you will get your appID and appKey and put their values into App.config in the application provided. The solution contains the SDK itself and a sample application that contains 2 sample use cases (one with intent routers and one using the client directly)

[iOS SDK for the Microsoft Face API](https://github.com/Microsoft/Cognitive-Face-iOS) This repo contains the iOS client library & sample for the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[Cognitive Services Face client library for Android](https://github.com/Microsoft/Cognitive-Face-Android) Cognitive Services Face client library for Android. https://www.microsoft.com/cognitive-services/en-us/face-api - This repo contains the Android client library & sample for the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[Windows SDK for the Microsoft Face API](https://github.com/Microsoft/Cognitive-Face-Windows) This repo contains the Windows client library & sample for the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[Python SDK for the Microsoft Face API](https://github.com/Microsoft/Cognitive-Face-Python) This Jupyter Notebook demonstrates how to use Python with the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[Python SDK for the Microsoft Speaker Recognition API](https://github.com/Microsoft/Cognitive-SpeakerRecognition-Python) This repo contains Python samples (using Python 3) to demonstrate the use of Microsoft Speaker Recognition API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[C++ REST SDK](https://github.com/Microsoft/cpprestsdk) The C++ REST SDK is a Microsoft project for cloud-based client-server communication in native code using a modern asynchronous C++ API design. This project aims to help C++ developers connect to and interact with services. Are you new to the C++ Rest SDK? To get going we recommend you start by taking a look at our tutorial to use the http_client. It walks through how to setup a project to use the C++ Rest SDK and make a basic Http request. Other important information, like how to build the C++ Rest SDK from source, can be located on the documentation page. 

[VIPR: Client Library Generation Toolkit](https://github.com/Microsoft/Vipr) VIPR is an extensible toolkit for generating Web Service Client Libraries. VIPR is designed to be highly extensible, enabling developers to adapt it to read new Web Service description languages and to create libraries for new target platforms with ease.  This repository contains the core VIPR infrastructure, Readers for OData v3 and v4, and Writers for C#, Objective-C, and Java. It also contains a Windows Command Line Interface application that can be used to drive Client Library generation.

## Interface

[thrifty](https://github.com/Microsoft/thrifty) Thrifty is an implementation of the Apache Thrift software stack for Android, which uses 1/4 of the method count taken by the Apache Thrift compiler. Thrift is a widely-used cross-language service-definition software stack, with a nifty interface definition language from which to generate types and RPC implementations. Unfortunately for Android devs, the canonical implementation generates very verbose and method-heavy Java code, in a manner that is not very Proguard-friendly. Like Square's Wire project for Protocol Buffers, Thrifty does away with getters and setters (and is-setters and set-is-setters) in favor of public final fields. It maintains some core abstractions like Transport and Protocol, but saves on methods by dispensing with Factories and server implementations and only generating code for the protocols you actually need. Thrifty was born in the Outlook for Android codebase; before Thrifty, generated thrift classes consumed 20,000 methods. After Thrifty, the thrift method count dropped to 5,000.

[Bond](https://github.com/Microsoft/bond) Bond is a cross-platform framework for working with schematized data. It supports cross-language de/serialization and powerful generic mechanisms for efficiently manipulating data. Bond is broadly used at Microsoft in high scale services. Bond is an open source, cross-platform framework for working with schematized data. It supports cross-language serialization/deserialization and powerful generic mechanisms for efficiently manipulating data. Bond is broadly used at Microsoft in high scale services.  We are also introducing the Bond Communications framework--known as Bond Comm--which allows for remote process communication. Currently, we are making the C# version of this framework available; the C++ version will be released in the coming weeks. This framework is based on is the successor to an internal framework that is used by several large services inside Microsoft. Bond Comm is undergoing active evolution at this time and so we are marking the initial release as version 0.5. Consult the C# manual for more details on Bond Comm's usage and capabilities. [Bond Comm C# Example](https://github.com/Microsoft/bond-comm-cs-example) A simple example service, demonstrating the Communications framework of Bond. There are two examples in this repository, demonstrating the server- and client-side of a simple calculator.



## Distributed Communication

[Robust Distributed System Nucleus (rDSN)](https://github.com/Microsoft/rDSN) Robust Distributed System Nucleus (rDSN) is a framework for quickly building robust distributed systems. It has a microkernel for pluggable components, including applications, distributed frameworks, devops tools, and local runtime/resource providers, enabling their independent development and seamless integration. The project was originally developed for Microsoft Bing, and now has been adopted in production both inside and outside Microsoft. 

## Languages

The ML world has many complex systems originally constructed in a variety of programming languages. Language interoperability and tool support becomes an important issue

[Visual F# compiler and tools](https://github.com/Microsoft/visualfsharp) F# is a mature, open source, cross-platform, functional-first programming language which empowers users and organizations to tackle complex computing problems with simple, maintainable, and robust code. F# is used in a wide range of application areas and is supported by Microsoft and other industry-leading companies providing professional tools, and by an active open community. You can find out more about F# at http://fsharp.org.

[R Tools for Visual Studio](https://github.com/Microsoft/RTVS) R Tools for Visual Studio (RTVS). 

[Node.js Tools for Visual Studio](https://github.com/Microsoft/nodejstools) NTVS is a free, open source plugin that turns Visual Studio into a Node.js IDE. It is designed, developed, and supported by Microsoft and the community.

[Python Tools for Visual Studio](https://github.com/Microsoft/PTVS) PTVS is a free, open source plugin that turns Visual Studio into a Python IDE. 

## Deployment and DevOps

[PowerShell Module for Docker](https://github.com/Microsoft/Docker-PowerShell) This repo contains a PowerShell module for the Docker Engine. It can be used as an alternative to the Docker command-line interface (docker), or along side it. It can target a Docker daemon running on any operating system that supports Docker, including both Windows and Linux.





[CommandLine-Documentation](https://github.com/Microsoft/CommandLine-Documentation) Gathering markdown documentation for Microsoft Command Line and associated interpreters.


[Bash on Ubuntu on Windows](https://github.com/Microsoft/BashOnWindows)

-    Documentation: https://msdn.microsoft.com/en-us/commandline/wsl/about
-    Release Notes: https://msdn.microsoft.com/en-us/commandline/wsl/release_notes
-    User Voice: https://wpdev.uservoice.com/forums/266908-command-prompt-console-bash-on-ubuntu-on-windo/category/161892-bash
-    WSL Blog: https://blogs.msdn.microsoft.com/wsl
-    Console Blog: https://blogs.msdn.microsoft.com/commandline/
-    Stack Overflow: https://stackoverflow.com/questions/tagged/wsl
-    List of programs that work and don't work: https://github.com/ethanhs/WSL-Programs
-    Discussion forum: http://wsl-forum.qztc.io
-    Tips and guides for new bash users: https://github.com/abergs/ubuntuonwindows





[PowerShell Desired State Configuration - for Linux](https://github.com/Microsoft/PowerShell-DSC-for-Linux) Supported Linux operating systems - The following Linux operating system versions are supported by DSC for Linux.

    CentOS 5, 6, and 7 (x86/x64)
    Debian GNU/Linux 6, 7 and 8 (x86/x64)
    Oracle Linux 5, 6 and 7 (x86/x64)
    Red Hat Enterprise Linux Server 5, 6 and 7 (x86/x64)
    SUSE Linux Enterprise Server 10, 11 and 12 (x86/x64)
    Ubuntu Server 12.04 LTS, 14.04 LTS, 16.04 LTS (x86/x64)


[Yeoman generator for Docker](https://github.com/Microsoft/generator-docker) This generator creates a Dockerfile and scripts (dockerTask.sh and dockerTask.ps1) that helps you build and run your project inside of a Docker container. The following project types are currently supported:

[Run Hadoop Cluster within Docker Containers](https://github.com/kiwenlau/hadoop-cluster-docker) By packaging Hadoop into Docker image, we can easily build a Hadoop cluster within Docker containers on local host. http://kiwenlau.com/2016/06/26/hadoop-cluster-docker-update-english/

[HDFS/Spark Workbench](https://github.com/big-data-europe/docker-hadoop-spark-workbench) This repo includes deployment instructions for running HDFS/Spark inside docker containers. Also includes spark-notebook and HDFS FileBrowser.


# Appendix - IOT Notes

- [Spending on IoT predicted to reach $1.7 trillion by 2020](http://venturebeat.com/2015/06/03/spending-on-iot-predicted-to-reach-1-7-trillion-by-2020/)
- [Google starts early access program to let developers try its Brillo OS for connected devices](http://venturebeat.com/2015/10/27/google-starts-early-access-program-to-let-developers-try-its-brillo-os-for-connected-devices/) “Brillo brings the simplicity and speed of software development to hardware by offering you a lightweight embedded OS based on Android, core services, a developer kit, and a developer console,” Googlers Gayathri Rajan and Ryan Cairns wrote in a blog post today. “You can choose from a variety of hardware capabilities and customization options, quickly move from prototype to production, and manage at scale with over the air (OTA) updates, metrics, and crash reporting.” Brillo works with Intel, MIPS, and ARM-based chips, but for now Google is steering people toward Brillo-certified boards. The OS gets small updates every six weeks
- [Microsoft releases free Windows 10 IoT Core for Raspberry Pi 2, MinnowBoard Max](http://venturebeat.com/2015/08/10/microsoft-releases-free-windows-10-iot-core-for-raspberry-pi-2-minnowboard-max/) Windows 10 IoT Core (the tiny version of Windows designed for sensor-laden Internet-connected devices) for two types of maker-friendly hardware: the Raspberry Pi 2 and the MinnowBoard Max. Wi-Fi and Bluetooth support, as well as improved GPIO performance on the Raspberry Pi 2
- [NXP unveils a tiny 64-bit ARM processor for the Internet of Things](http://venturebeat.com/2016/02/21/nxp-unveils-a-small-and-tiny-64-bit-arm-processor-for-the-internet-of-things/) NXP Semiconductors has unveiled what it calls the world’s smallest and lowest-power 64-bit ARM processor for the Internet of Things (IoT). The tiny QorIQ LS1012A delivers networking-grade security and performance acceleration to battery-powered, space-constrained applications. This includes powering applications for Internet of Things, or everyday objects that are smart and connected. If IoT is to reach its potential of $1.7 trillion by 2020 (as estimated by market researcher IDC), it’s going to need processors like the new one from NXP, which was unveiled at the Embedded World 2016 event in Nuremberg, Germany. The chip has a 64-bit ARMv8 processor with network packet acceleration and built-in security. It fits in a 9.6 mm-square space and draws about 1 watt of power. Potential applications include next-generation IoT gateways, portable entertainment platforms, high-performance portable storage applications, mobile hard disk drives, and mobile storage for cameras, tablets, and other rechargeable devices.
- [That’s not a blade of grass — It’s a Freescale Internet of Things chip](http://venturebeat.com/2015/12/01/thats-not-a-blade-of-grass-its-a-freescale-internet-of-things-chip-2/) Freescale‘s newest chip is as thin as a blade of grass. Targeted at Internet of Things applications, which are expected to become a $1.7 trillion market by 2020, the Kinetis K22 microcontroller from Freescale is just 0.34 millimeters in height. But it packs a 120-megahertz processor and a variety of memories and interfaces into a tiny little package for Internet of Things applications. The Kinetis is a new breed of microcontroller, or MCU, which packs all of the necessary components for running an appliance-like device, such as an automated glucose monitor for diabetes patients. Freescale envisions its chip being embedded in a stretchable electronic patch or even under the skin, as an implant for such monitors. Earlier this year, Freescale unveiled the world’s smallest single-chip module (SCM) for the IoT, replacing a six-inch board with a device the size of the U.S. dime and reducing the need for 100-plus components down to just one.
- [Come on! You know you want to connect your wall sockets and light bulbs to the Internet](http://venturebeat.com/2016/01/05/you-know-you-want-to-connect-your-wall-sockets-and-light-bulbs-to-the-internet/) The iDevices Wall Outlet gives you the ability to control power to an outlet. It will be ready for sale in the third quarter. No word on pricing just yet.
- [Y Combinator-backed Mosaic connects wearables, Internet of Things](http://venturebeat.com/2016/07/12/y-combinator-backed-mosaic-connects-wearables-internet-of-things/) has released a new beta version of its assistant on Amazon Alexa, Slack, SMS, and Facebook Messenger. Mosaic connects a single chatbot to more than a dozen wearables and IoT devices, ranging from lightbulbs to Tesla cars. Essentially, the chatbot takes all the data we compile in our personal lives and helps us make sense of it. Mosaic wants to give people advice based on all the data accumulated by wearables and internet-connected devices and can also explain things like water usage and how to get energy. In the future, Mosaic wants your fitness wearable to talk to you about personal fitness challenges, make sure you’re getting enough sleep, and tell you when you need to exercise
- [Microsoft, Intel, Samsung, & others launch IoT standards group: Open Connectivity Foundation](http://venturebeat.com/2016/02/19/microsoft-intel-samsung-others-launch-iot-standards-group-open-connectivity-foundation/) Giants of the tech world are banding together to found a new group to support the burgeoning Internet of Things (IoT) industry. The Open Connectivity Foundation (OCF) is touted as an open IoT standards group to unify standards, expedite innovation, and “create IoT solutions and devices that work seamlessly together,” according to a press release. Founding members include Microsoft, Cisco, Electrolux, General Electric, Intel, Qualcomm, Samsung, ARRIS, and CableLabs, who will work together to create specifications and protocols to ensure devices from a myriad of manufacturers work in harmony. Elsewhere, Microsoft is making a big IoT play with Windows 10 and Azure, as the company looks to build an operating system that delivers access to universal apps and driver models that work across any device, from fridges and ATMs to industrial robotics.
- [Teens use Windows 10 IoT Core to run science experiments in space](http://venturebeat.com/2016/03/31/teens-use-windows-10-iot-core-to-run-science-experiments-in-space/)  they’ll conduct science experiments on things like seeing how metals react to electromagnetic energy in space. They’ll rely on a robotic arm and a camera to take photographic evidence of what happens. The code running on the board is written in C#. What this shield did is it gave us the ability to have eight copies of the OS in eight different USB sticks,” Quest’s Kim said. “They created a custom hardware watchdog to see if the OS got corrupted … it would actually reboot to the next uncorrupted OS. It gave us a system to operate in the harsh environment of space.”
- [How Silicon Valley is botching IoT](http://venturebeat.com/2016/05/22/how-silicon-valley-is-botching-iot/) The chief culprit here is not coding but culture. In Silicon Valley, the priority is to get on the latest disruptive platform and rush to be first to market. And we seem to be collectively suffering from amnesia. We keep seeing the same security problems over and over again in the mobile ecosystem, as inexperienced teams rush their apps to market, leaving many of them vulnerable to hacking. Repeating this pattern, we regularly see IoT devices being produced by people with little or no hardware experience and scant background dealing with interaction between hardware, middleware, and software.
- [Helium raises $20 million for smart industrial sensors](http://venturebeat.com/2016/04/25/helium-raises-20m-for-smart-industrial-sensors/) The Helium Green sensor can monitor temperature, humidity, barometric pressure, light, and motion. Helium also has its Helium Pulse application for the Web and for mobile devices. It enables remote monitoring and alerts so that companies can control Helium smart sensors, program alert parameters, and take other actions based on those insights. 
- [For the Internet of things, the cost of cheap will be steep](http://venturebeat.com/2015/01/10/for-the-internet-of-things-the-cost-of-cheap-will-be-steep/) But at the Black Hat conference this past year, hackers compromised the Google Nest thermostat to reveal the weaknesses of these connected devices and appliances. As the IoT market matures, these widely deployed and low-cost sensors and devices are less likely to be viewed as worth continued maintenance. Offering a constant stream of security patches and updates to keep low-cost devices safe and functional for the long term requires money. If vulnerabilities are discovered, patches or updates might be issued, but only in the first year or two. The vendor expectation is that users will need to buy a full replacement or live with the risks — not to mention that users are not very likely to manage patches and updates for non-critical devices. Cheap and vulnerable devices will linger on networks like ticking time bombs, and the choice will be to either replace them or tolerate them with their liabilities. Simply tolerating the risks of low-cost devices could incite major long-term challenges for our economy.
- [Verdigris raises $6.7 million for artificial intelligence that powers green factories and hotels](http://venturebeat.com/2016/10/20/verdigris-raises-6-7-million-for-artificial-intelligence-that-powers-green-factories-and-hotels/) The smart energy startup Verdigris announced today that it has raised $6.7 million to scale production of its Einstein smart sensor and frequency detectors. The sensors are used to predict the failure of machines and improve energy efficiency.
- [Xenio raises funding for smart lighting and the Internet of Things (update)](http://venturebeat.com/2016/10/06/xenio-raises-5-million-for-smart-lighting-and-the-internet-of-things/) The company believes it can accelerate the deployment of the Internet of Things by embedding its cloud-based solution into Bridgelux’s smart lights, which are controlled via internet apps. Xenio uses those lights as beacons that can discover who is nearby and beam location-based promotional messages to the passersby. Xenio collects data through a two-way communications loop with smartphone users, and it provides them with targeted marketing messages from the businesses the customers are already visiting.
- [IBM to pour $200 million into Watson Internet of Things A.I. business in Munich](http://venturebeat.com/2016/10/03/ibm-to-pour-200-million-into-watson-internet-of-things-a-i-business-in-munich/) It’s also part of a global plan to invest $3 billion to bring Watson’s cognitive computing to IoT. The investment in Munich is one of the company’s largest ever in Europe and is in response to growing demand from customers who want to transform their operations with A.I. and IoT.
- [Beyond Verbal wants to use virtual assistants to detect disease by analyzing your voice](http://venturebeat.com/2016/09/26/beyond-verbal-wants-to-use-virtual-assistants-to-detect-disease-by-analyzing-your-voice/) For the past two years, a startup called Beyond Verbal has been working on disease detection through voice samples and machine learning, an application Zuckerberg himself has talked about as having interesting potential. Today, the company launched the Beyond mHealth Research Platform to collaborate with research institutes, hospitals, businesses, and universities to collectively search for unique markers in voice samples.
- [Amazon’s Alexa Fund leads $5.6 million round for smart home startup Nucleus](http://venturebeat.com/2016/09/21/amazons-alexa-fund-leads-5-6-million-round-for-smart-home-startup-nucleus/) Enabled by Wi-Fi, Nucleus lets you use voice or video in the home to chat with other Nucleus devices or with Nucleus iOS and Android apps around the world
- [GE and Cisco face off over industrial IoT](http://venturebeat.com/2016/08/27/ge-and-cisco-face-off-over-industrial-iot/) Corporate giants are in an all-out race to dominate the widely coveted Internet of Things (IoT) landscape. Today, eight companies with market capitalizations over $150 billion actively promote IoT solutions: Google, Microsoft, Amazon, General Electric, AT&T, Verizon, IBM, and Cisco. When you take away the cloud, computing, and telecommunications players, you’re left with an interesting face off between General Electric and Cisco — both of which are targeting industrial IoT. U.S. organizations will invest more than $232 billion in Internet of Things hardware, software, services, and connectivity this year. And IDC expects U.S. IoT revenues will experience a compound annual growth rate of 16.1 percent over the 2015-2019 forecast period, reaching more than $357 billion in 2019.
- [How Intel and GE plan to make cities smarter](http://venturebeat.com/2016/08/17/how-intel-and-ge-team-plan-to-make-cities-smarter/) Intel estimates that each one of us will use 1.5 gigabytes of data per day by 2020. A smart hospital will use 3,000 gigabytes a day. And a smart factory could use a million gigabytes of data per day. “We are at a line of demarcation where you embrace the future or you are unable to satisfy the needs of your customers,” said Immelt. “Every industrial company has to transform itself into a digital company.”
- [Xperiel raises $7 million to connect the digital and real worlds with mixed reality](http://venturebeat.com/2016/08/11/xperiel-raises-7-million-to-connect-the-digital-and-real-worlds-with-mixed-reality/) Technology today is largely inadequate when it comes to addressing how to leverage the IoT and physical infrastructure to better engage customers and drive commerce,” said Stephen Hendrick, principal analyst for application development and deployment research at ESG, in a statement. “Xperiel is breaking new ground by providing a highly abstracted language for building event-driven, device-agnostic applications, as well as an IoT-centric connectivity fabric that ties together devices, events, and content to enable customer-engagement and commerce that is far more effective than other available solutions.”
- [Onboard diagnostics will connect cars to the Internet of Things](http://venturebeat.com/2016/08/07/onboard-diagnostics-will-connect-cars-to-the-internet-of-things/) A 2011 Machina Research study forecasted that by 2022, there will be 1.8 billion machine-to-machine (M2M) automotive connections, consisting of 700 million connected cars and a $1.1 billion aftermarket in devices for services.
- [Internet of random things: Fridges, doorbells, beds, and ovens are your connected future. Right?](http://venturebeat.com/2016/03/26/internet-of-random-things-fridges-doorbells-beds-and-ovens-are-your-connected-future-right/) San Francisco-based June announced a fresh $22.5 million funding round, taking its total money raised to the $30 million mark to help bring its smart oven to market. The June oven isn’t like any ordinary oven — this one sports a camera inside that uses deep learning techniques to figure out what you’re preparing, then sets the appropriate temperature to ensure your meal is adequately cooked. The news comes two weeks after newcomer and Y Combinator-backed Tovala launched a Kickstarter for its smart oven, a campaign that’s currently sitting at more than 200 percent of its target funding.
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()

# Appendix - Other
- [Nvidia sees government as its next A.I. goldmine](http://venturebeat.com/2016/10/23/nvidia-ai-government/)
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()



- []()

starting to improve IOT and visualization-3








