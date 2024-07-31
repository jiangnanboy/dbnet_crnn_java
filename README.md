<h3 align="center">dbnet_crnn_java</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
          <a href="#result">Result</a>
        </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>

#### About

- 本项目利用java,javacv,onnx以及djl矩阵计算等技术加载文本检测模型dbnet与文本识别模型crnn，完成ocr的识别推理。
- 包含模型的完整项目请从右侧releases处下载。

#### Getting started
目前只开放了通用文本检测与通用文本识别模型。【另有印刷体文档识别、手写体文字识别以及自然场景文字识别模型未开放，如有需求可联系我。】
- DBNetTest.java: 通用文本检测
- CRNNTest.java:通用文本识别
- DetRecTest.java:通用文本检测和文本识别

#### Result
- 通用文本检测结果展示：
<br/><br/> 
<p align="center">
  <a>
    <img src="imgs/test_imgs/det1.jpg">
  </a>
</p>
<br/><br/>

<br/><br/> 
<p align="center">
  <a>
    <img src="imgs/test_imgs/det2.jpg">
  </a>
</p>
<br/><br/>

<br/><br/> 
<p align="center">
  <a>
    <img src="imgs/test_imgs/det3.jpg">
  </a>
</p>
<br/><br/>

- 通用文本识别结果展示：

以下是imgs/det/test1.jpg的识别结果：
 ```
    text: 2018人工智能未来企业排行榜
    text: 领域
    text: 排名
    text: 企业
    text: 百度
    text: 开放的人工智能服务平台
    text: 1
    text: 2
    text: 腾讯
    text: 互联网综合服务
    text: 华为
    text: 3
    text: 人工智能自动化业务、智能芯片
    text: 阿里巴巴
    text: 互联网综合服务
    text: 4
    text: 5
    text: 平安集团
    text: 人工智能金融研发平台
    text: 6
    text: 华大基因
    text: 精准检测、医疗数据运营服务
    text: 7
    text: 搜狗
    text: 综合人工智能解决方案平台
    text: 8
    text: 科大讯飞
    text: 智能语音技术
    text: 9
    text: 中科创达
    text: 智能终端平台技术
    text: 10
    text: 珍岛集团
    text: SaaS级智能营销云平台
    text: I
    text: 商汤科技
    text: 人工智能视觉深度学习平台
    text: 12
    text: 神州泰岳
    text: 综合类软件产品及服务
    text: 13
    text: 寒武纪科技
    text: 深度学习专用的智能芯片
    text: 14
    text: 汉王科技
    text: 文字识别技术与智能交互
    text: 15
    text: 全志科技
    text: 智能芯片设计
    text: 16
    text: facc++旷视科技
    text: 人工智能产品和行业解决方案
    text: 17
    text: 创略科技
    text: 智能客户数据平台
    text: 18
    text: 海云数据
    text: 企业级大数据整体运营与分析服务
    text: 19
    text: 影谱科技
    text: 视觉技术、智能影像生产企业
    text: 20
    text: 智臻智能
    text: 智能机器人技术提供和平台运营
 ```
#### Contact
如有问题，联系我：

1、github：https://github.com/jiangnanboy

2、QQ:2229029156



