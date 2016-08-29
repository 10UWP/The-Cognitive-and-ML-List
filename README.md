# :chart: The-Cognitive-and-ML-List

Cognitive Services, ML, Machine Learning, Data Ingestion, Azure, Docker, Container, R, Python, C#, Java, Hadoop, Spark etc

[Documentation for Microsoft Cognitive Services](https://github.com/Microsoft/Cognitive-Documentation) This repo contains the documentaiton for Microsoft Cognitive Services, which includes the former Project Oxford APIs. You can also check out our SDKs & Samples on our website.  Don't see what you're looking for? We're working on expanding our offerings and would love to hear from you what APIs, docs, SDKs, and samples you want to see next. Let us know on the Cognitive Services UserVoice Forum.

[Computational Network Toolkit (CNTK)](https://github.com/Microsoft/CNTK) CNTK (http://www.cntk.ai/), the Computational Network Toolkit by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

[Language Understanding Intelligent Service API](https://github.com/Microsoft/Cognitive-LUIS-Windows) Windows (.Net) SDK for the Microsoft Language Understanding Intelligent Service API, part of Congitive Services http://www.microsoft.com/cognitive-services/en-us/language-understanding-intelligent-service-luis - LUIS is a service for language understanding that provides intent classification and entity extraction. In order to use the SDK you first need to create and publish an app on www.luis.ai where you will get your appID and appKey and put their values into App.config in the application provided. The solution contains the SDK itself and a sample application that contains 2 sample use cases (one with intent routers and one using the client directly)

[Mobius](https://github.com/Microsoft/Mobius) C# language binding and extensions to Apache Spark. Mobius provides C# language binding to Apache Spark enabling the implementation of Spark driver program and data processing operations in the languages supported in the .NET framework like C# or F#.

[Cognitive Services Face client library for Android](https://github.com/Microsoft/Cognitive-Face-Android) Cognitive Services Face client library for Android. https://www.microsoft.com/cognitive-services/en-us/face-api - This repo contains the Android client library & sample for the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[Windows SDK for the Microsoft Face API](https://github.com/Microsoft/Cognitive-Face-Windows) This repo contains the Windows client library & sample for the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[thrifty](https://github.com/Microsoft/thrifty) Thrifty is an implementation of the Apache Thrift software stack for Android, which uses 1/4 of the method count taken by the Apache Thrift compiler. Thrift is a widely-used cross-language service-definition software stack, with a nifty interface definition language from which to generate types and RPC implementations. Unfortunately for Android devs, the canonical implementation generates very verbose and method-heavy Java code, in a manner that is not very Proguard-friendly. Like Square's Wire project for Protocol Buffers, Thrifty does away with getters and setters (and is-setters and set-is-setters) in favor of public final fields. It maintains some core abstractions like Transport and Protocol, but saves on methods by dispensing with Factories and server implementations and only generating code for the protocols you actually need. Thrifty was born in the Outlook for Android codebase; before Thrifty, generated thrift classes consumed 20,000 methods. After Thrifty, the thrift method count dropped to 5,000.

[Multiverso](https://github.com/Microsoft/multiverso) Multiverso is a parameter server based framework for training machine learning models on big data with numbers of machines. It is currently a standard C++ library and provides a series of friendly programming interfaces. With such easy-to-use APIs, machine learning researchers and practitioners do not need to worry about the system routine issues such as distributed model storage and operation, inter-process and inter-thread communication, multi-threading management, and so on. Instead, they are able to focus on the core machine learning logics: data, model, and training. For more details, please view our website http://www.dmtk.io.

[Visual F# compiler and tools](https://github.com/Microsoft/visualfsharp) F# is a mature, open source, cross-platform, functional-first programming language which empowers users and organizations to tackle complex computing problems with simple, maintainable, and robust code. F# is used in a wide range of application areas and is supported by Microsoft and other industry-leading companies providing professional tools, and by an active open community. You can find out more about F# at http://fsharp.org.

[Bot Builder SDK](https://github.com/Microsoft/BotBuilder) The Microsoft Bot Builder SDK is one of three main components of the Microsoft Bot Framework. The Microsoft Bot Framework provides just what you need to build and connect intelligent bots that interact naturally wherever your users are talking, from text/SMS to Skype, Slack, Office 365 mail and other popular services. http://botframework.com - Bots (or conversation agents) are rapidly becoming an integral part of one’s digital experience – they are as vital a way for users to interact with a service or application as is a web site or a mobile experience. Developers writing bots all face the same problems: bots require basic I/O; they must have language and dialog skills; and they must connect to users – preferably in any conversation experience and language the user chooses. The Bot Framework provides tools to easily solve these problems and more for developers e.g., automatic translation to more than 30 languages, user and conversation state management, debugging tools, an embeddable web chat control and a way for users to discover, try, and add bots to the conversation experiences they love.

[PowerShell Module for Docker](https://github.com/Microsoft/Docker-PowerShell) This repo contains a PowerShell module for the Docker Engine. It can be used as an alternative to the Docker command-line interface (docker), or along side it. It can target a Docker daemon running on any operating system that supports Docker, including both Windows and Linux.

[R Tools for Visual Studio](https://github.com/Microsoft/RTVS) R Tools for Visual Studio (RTVS). 

[Node.js Tools for Visual Studio](https://github.com/Microsoft/nodejstools) NTVS is a free, open source plugin that turns Visual Studio into a Node.js IDE. It is designed, developed, and supported by Microsoft and the community.

[Multiworld Testing Decision Service](https://github.com/Microsoft/mwt-ds) https://github.com/Microsoft/mwt-ds/wiki [MWT White Paper: Methodology; system design; how to make the DS fit your application; past deployments; experimental evaluation](https://github.com/Microsoft/mwt-ds/raw/master/images/MWT-WhitePaper.pdf)

    How to tailor the DS to YOUR application (conceptually, in terms of machine learning)
    How to track competing predictions in real time.
    How to extract your model for experimentation outside of the DS
    How to delete your instance to save money, conserve your quota, etc.

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

[VIPR: Client Library Generation Toolkit](https://github.com/Microsoft/Vipr) VIPR is an extensible toolkit for generating Web Service Client Libraries. VIPR is designed to be highly extensible, enabling developers to adapt it to read new Web Service description languages and to create libraries for new target platforms with ease.  This repository contains the core VIPR infrastructure, Readers for OData v3 and v4, and Writers for C#, Objective-C, and Java. It also contains a Windows Command Line Interface application that can be used to drive Client Library generation.

[Python Tools for Visual Studio](https://github.com/Microsoft/PTVS) PTVS is a free, open source plugin that turns Visual Studio into a Python IDE. 

[PowerShell Desired State Configuration - for Linux](https://github.com/Microsoft/PowerShell-DSC-for-Linux) Supported Linux operating systems - The following Linux operating system versions are supported by DSC for Linux.

    CentOS 5, 6, and 7 (x86/x64)
    Debian GNU/Linux 6, 7 and 8 (x86/x64)
    Oracle Linux 5, 6 and 7 (x86/x64)
    Red Hat Enterprise Linux Server 5, 6 and 7 (x86/x64)
    SUSE Linux Enterprise Server 10, 11 and 12 (x86/x64)
    Ubuntu Server 12.04 LTS, 14.04 LTS, 16.04 LTS (x86/x64)

[Project Malmo](https://github.com/Microsoft/malmo) Project Malmo is a platform for Artificial Intelligence experimentation and research built on top of Minecraft. We aim to inspire a new generation of research into challenging new problems presented by this unique environment. --- For installation instructions, scroll down to *Getting Started* below, or visit the project page for more information: https://www.microsoft.com/en-us/research/project/project-malmo/

[Python SDK for the Microsoft Face API](https://github.com/Microsoft/Cognitive-Face-Python) This Jupyter Notebook demonstrates how to use Python with the Microsoft Face API, an offering within Microsoft Cognitive Services, formerly known as Project Oxford.

[C++ REST SDK](https://github.com/Microsoft/cpprestsdk) The C++ REST SDK is a Microsoft project for cloud-based client-server communication in native code using a modern asynchronous C++ API design. This project aims to help C++ developers connect to and interact with services. Are you new to the C++ Rest SDK? To get going we recommend you start by taking a look at our tutorial to use the http_client. It walks through how to setup a project to use the C++ Rest SDK and make a basic Http request. Other important information, like how to build the C++ Rest SDK from source, can be located on the documentation page. 

[HoloToolkit](https://github.com/Microsoft/HoloToolkit) The HoloToolkit is a collection of scripts and components intended to accelerate the development of holographic applications targeting Windows Holographic.








