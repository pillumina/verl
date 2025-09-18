我希望你对当前仓库这个训练框架进行分析调研：

调研重点是在于这个框架的训练核心模块功能是用了那些三方库能力，举个例子，比如verl框架依赖了tensordict，ray等库。我的要求是：
1. 需要分析业界开源关键的训练基础库，比如tensordict这类和框架数据结构构建有关的，ray这种和分布式构建有关的，triton、fa等等，那些工具类的依赖不要分析。
2. 将训练框架分成几个关键的模块，比如attention、data loader等这类的，当然需要根据每个框架的实际情况来。
3. 将分析后得到的每个基础库和框架的功能模块进行关联（连线箭头表示）

举个之前分析推理加速库的例子，分析vllm和sglang的：
- sglang有115个triton算子（41文件），44个cu文件；vllm有73个triton算子(27文件)，131个cu文件，FA大量调用业界加速库
- 加速库聚焦高性能API，完备性非关键，FlashInfer主要提供CUDA C实现的高性能API，由Triton实现的仅4个，且都有高性能实现。
- 可以把vllm分成几块：Attention，sample, MLA, quant, moe, lora；把sglang分成：Attention，sample，MLA,quant, moe, lora, spec。其中关键的Triton基础组件，和两个推理框架的所有模块都相关；FlashAttention基础组件，和两个框架的Attention模块相关；DeepGEMM和sglang的quant和moe相关；pytorch的FlexAttention和vllm的Attention相关；等等这些关联可以形成关联的网络用图表现。

输出的要求:
1. 输出格式为markdown文件
2. 输出文件名为{框架名}-deps-analysis.md。
3. 除了文字描述以外，关联关系还要以mermaid或者gantt图这种形式进行表现。
4. 分析最好带上信源。
   