#import "../../config.typ": *
#import "@preview/cetz:0.4.0": canvas, draw
#import "./section2_plot.typ": *
== 逻辑门

#v1
#h2 这一章将视角从拟合上短暂地移开，我相信理解逻辑和数据的关系多少也会帮助我们理解神经网络。读者或许好奇过，计算是如何完成的呢？在讨论这个问题之前，先来做一个约定，我们将 0 视作#textOverSet("False", "假")，1 视作#textOverSet("True", "真")#footnote[逻辑0/1：在物理上，逻辑 0 由#textOverSet("Low", "低电平")表示，逻辑 1 由#textOverSet("High", "高电平")表示，TTL 和 CMOS 电路各有多种电压标准，感兴趣的读者可以自行学习电路的知识。]。先来看看几种最简单的逻辑运算。

1. #textOverSet("Not", "非")（数学写法：$not x$，C 语言写法：`!`，Python 写法：`not`）

#h2 非是一元运算符，它只有一个输入，输出与输入相反，其中
   $
       not 0 = 1, not 1 = 0
   $
   也就是说 $not x = 1 - x$，$x$ 与 $not x$ 是互补的。如果你看逻辑 0, 1 仍然感觉不太自然，你可以把它想成 False = not True, True = not False。

2. #textOverSet("And", "与")（数学写法：$and$，C 语言写法：`&&`，Python 写法：`and`）

#h2 与是二元运算符，它有两个输入，仅当两输入都为 1 时输出为 1，否则输出为 0，从真值表#footnote[真值表：逻辑运算的输出与输入的关系表。]就可以看出这一点：

#figure(
  table(
    columns: 3,
    align: center,

    stroke: (x, y) => {
      if y==0 {
        (bottom: 0.7pt, top: 0.7pt)
      } 
      if y==4 {
        (bottom: 0.7pt)
      }
    },
    [$x$], [$y$], [$x and y$],
    [0], [0], [0],
    [0], [1], [0],
    [1], [0], [0],
    [1], [1], [1]
  ),
  caption: [与的真值表]
)

这与乘法的结果是一样的，所以有时也会省去和的符号，使用 $x y$ 表示 $x and y$。

3. #textOverSet("Or", "或")（数学写法：$or$，C 语言写法：`||`，Python 写法：`or`）

#h2 或是二元运算符，它有两个输入，仅当两输入都为 0 时输出为 0，否则输出为 1，真值表如下：

#figure(
  table(
    columns: 3,
    align: center,

    stroke: (x, y) => {
      if y==0 {
        (bottom: 0.7pt, top: 0.7pt)
      } 
      if y==4 {
        (bottom: 0.7pt)
      }
    },

    [$x$], [$y$], [$x or y$],
    [0], [0], [0],
    [0], [1], [1],
    [1], [0], [1],
    [1], [1], [1]
  ),
  caption: [或的真值表]
)

在图上这些运算一般会这样表示：

#figure(
  grid(
    columns: 3,
    gutter: 1em,
    image("../../../img/not_gate.png", width: 60%),
    image("../../../img/and_gate.png", width: 60%),
    image("../../../img/or_gate.png", width: 60%)
  ),
  caption: [逻辑门：从左到右分别为非门、与门、或门]
)

#h2 看起来这只是一些非常简单的运算，但是有了这些就可以构建出所有的计算#footnote[所有的运算：这里指的是#textOverSet("Turing Completeness", "图灵完备性")，如果你想深入了解，可以在 Steam 上购买一个叫做 #link("https://store.steampowered.com/app/1444480/Turing_Complete/")[Turing Complete] 的硬核游戏，推荐游玩。]。例如#textOverSet("Exclusive Or", "异或")运算表示两个输入不同。最粗暴简单的定义方法是列出其输出为 1 的所有情况：$x "xor" y = (x and not y) or (not x and y)$。这样就可以用非、与、或门来构建出一个异或门。

虽然它可以完成"所有的运算"，但是具体来说，比如有读者可能要问，如果我想计算加法，它应该怎么办呢？既然逻辑上只有两个值，那么自然地计算机就要使用二进制来表示数字了。二进制的加法非常简单，就以 $5+3$ 为例，我们可以这样计算：

#figure(
  table(
    columns:1,
    align: right,
    stroke: (x, y) => {
      if y==1{
        (bottom: 0.7pt)
      }
    },
    [101], [\+ #h(1em) 011], [1000]
  ),
  caption: [二进制加法]
)

逻辑门又是如何完成这一过程的呢？我们将它拆解成一个个小问题。当加到某一位时，我们需要考虑三个数：两个加数和来自后方的进位。例如下面这种情况：

#figure(
  table(
    columns: 4,
    align: left,
    stroke: (x, y) => {
      if y==1{
        (bottom: 0.7pt)
      }
    },
    [], [$...$], [1], [#text(gray)[$...$]],
    [\+],[$..._#text(red)[1]$], [$0_#text(blue)[1]$], [#text(gray)[$...$]],
    [], [$...$], [1], [#text(gray)[$...$]],
  ),
  caption: [二进制加法]
)

考虑这一位时，后面相加得到结果的情况我们已经不关心了（因此标为浅灰色），在这里只需关心从后方是否有进位（按照列竖式加法习惯，图中蓝色标注的下标 1）。再考虑两个加数的这一位分别为 1 和 0，所以 $1+0+1=2$，在结果栏写下一个 0（横线下方红色的 0），向前进位 1（写在前面一位下标的红色的 1），然后以同样的流程处理前一位。

记两个加数的这一位分别为 $x, y$，后方进位为 $c$，那么这一位的加和 $s$ 和向前进位 $c_n$ 可以表示为：

$ s = (x "xor" y) "xor" c $

$ c_n = (x and y) or (c and (x "xor" y)) $

#h2 当然这并不是唯一正确的写法，实际上有很多正确的写法，证明就免了，如果读者有兴趣可以自行尝试，或许也可以找到另外的表达式。最简单粗暴的方法就是把两个加数与是否带进位的情况全部列出来，分成 $2^3 = 8$ 种情况，就得到了如下的表，并逐一验证：

#figure(
  table(
  columns: (45pt, 45pt, 45pt, 45pt, 45pt, 45pt),
  align: center,
  rows: (5pt, auto, auto, auto, auto, auto, auto, auto, auto, auto), // 由于textOverSet太高了，需要用空的第一行来让它不会挤在顶部的分界线
  stroke: (x, y) => {
      if y==0{
        (top: 0.7pt)
      }
      if y==1{
        (bottom: 0.7pt)
      }
      if y==9{
        (bottom: 0.7pt)
      }
    },
  [],[],[],[],[],[],
  [#textOverSet($x$, "加数1")], [#textOverSet($y$, "加数2")], [#textOverSet($c$, "后方进位")], [#textOverSet("sum", "加和")], [#textOverSet($c'$, "向前进位")], [#textOverSet($s$, "结果位")],
  [0], [0], [0], [0], [0], [0],
  [0], [0], [1], [1], [0], [1],
  [0], [1], [0], [1], [0], [1],
  [0], [1], [1], [2], [1], [0],
  [1], [0], [0], [1], [0], [1],
  [1], [0], [1], [2], [1], [0],
  [1], [1], [0], [2], [1], [0],
  [1], [1], [1], [3], [1], [1]
))

#h2 就像等式描述的一样，每一个输出Y的位都可以通过输入的逻辑运算用一定的电路连接表示，把多个电路串起来#footnote[串起来：对于加法这个例子，在网上#link("https://www.bing.com/search?q=%E5%85%A8%E5%8A%A0%E5%99%A8", [搜索全加器])就可以很容易地搜到。]，就可以完成加法了。本质上我们的计算机 CPU 就是由这样的门电路与接线组成的#footnote[说明：实际上制造中，与非门、或非门使用更多，因为它们有更方便制造、体积较小、功耗低等优势。]。一个 CPU 需要大量门电路组合形成，现代的 CPU 包含数十亿个门电路，而一个门又由若干个微型的晶体管构成。为了让电路精确地实现我们预期的功能，需要精准地将电路雕刻在硅片上，这就是光刻技术如此重要的原因。但是山在那，总有人会去登的#footnote[山在那，总有人会去登：语出源自英国登山家 George Mallory 当被问及为何要攀登珠穆朗玛峰时的回答"因为山在那里"。‌刘慈欣的短篇小说#link("https://zhiqiang.org/resource/liucixin-mountain.html", [《山》])引用了这句话#footnote[约翰·肯尼迪在登月演讲中也引用了这句话。]。写到大量的微晶体管以精妙地排布构成电路让我想起小说中从基本电路开始进化的的硅基生物。如果你看到这里看累了，去看看小说放松一下吧。]，两个多世纪的技术积累才造就了现代计算机的诞生，从逻辑门到通用计算机每一步的发展都凝聚着人类技术与智慧的结晶。

#recommend(
  "推荐阅读",
[
如果你是 Minecraft 玩家或许见过使用红石电路制作的计算机，背后的原理可阅读：\
#h2 `计算器计算出「1+1=2」的整个计算过程是怎样的？为什么能秒算？ - WonderL的回答 - 知乎`\
#h2 #link("https://www.zhihu.com/question/29432827/answer/150408732")\
如果你有一些数字电路的基础，并想了解逻辑门是如何组合的，可以阅读：\
#h2 `计算器计算出「1+1=2」的整个计算过程是怎样的？为什么能秒算？ - Pulsar的回答 - 知乎`\
#h2 #link("https://www.zhihu.com/question/29432827/answer/150337041")\
]
)