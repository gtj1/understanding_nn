#import "../../config.typ": *
#import "section1_plot.typ": *

= 高维的线性拟合

#figure(
  image("../../../img/high_dimension_fitting.png", width: 60%),
  caption: [高维线性拟合示例]
)

#h2 第一节我们介绍了“简单线性回归”，即只有一个自变量的线性回归。但是在实际问题中，自变量往往不止一个，这时一元的线性回归就需要改成
#textOverSet("多元线性回归", "Multiple Linear Regression")。不过按照我的习惯，文中仍然称为“拟合”。

现实世界中的数据往往是多维的，就以估计体重为例，不难发现年龄和身高就是两个可能相关的变量。如果我们想用一个模型来描述这种相关性的话，最简单的就是线性模型了，与之前的 $hat(y)=k x+b$ 类似，自然想到用这样的函数
#footnote[记号说明：这里使用字母 $w$ 表示#textOverSet("权重", "weight")，$b$ 表示#textOverSet("偏置", "bias")，即常数项，$d$ 表示的是空间的#textOverSet("维度", "dimension")。]去拟合数据：

$
  hat(y) = w_1 x_1 + w_2 x_2 + dots + w_d x_d + b
$

同样地，优化的目标仍然是最小化均方误差，即对 $n$ 个数据点令
$
    "MSE" = 1/n sum_(i=1)^n (y_i - hat(y_i))^2
$

通过最小化误差得到 $w = [w_1, w_2, dots, w_d]$ 和 $b$ 的值。这个过程与一元线性回归的过程是类似的，只不过自变量是一维时，可以在平面上直接画出拟合的直线，二维时可以在空间中画出平面，但是当维数增加到三维及以上时，拟合所用的线性函数就变为#textOverSet("超平面", "Hyperplane")了，我们无法直观地看到这个超平面，但是可以猜测，它的原理差不多。

话不多说，先看看效果。这里以美国人类学家 Richard McElreath 搜集到的一个年龄、身高与体重#link("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv", "数据集")为例，它的分布与拟合出来的平面是这样的：

TODO!!!!

通过拟合，我们可以得到一个超平面，它大致描述了数据的分布。这个超平面的方程是 $0.04676645038926784 dot "age" + 0.47766688346191755 dot "height" - 31.805656676953056 = hat("weight")$，它比单纯使用身高或者年龄的拟合效果都要好一些。由此还可以量化地看到，年龄与身高都会影响体重，但是年龄是弱相关，而身高是强相关，这也符合我们的日常经验。

不过正如我们之前一直在做的一样，让我们看看更为直观的几何视角。仍然用 $bold(x^0)$ 表示全 $1$ 的向量，使用 $bold(upright(x))_(:1)$ 表示所有样本的第一个
#textOverSet("特征", "Feature")（分量），$bold(upright(x))_(:2)$ 表示所有样本的第二个特征，以此类推
#footnote[记号说明：冒号表示取所有行，这是为了与 Python 中 Numpy, Torch 等库的列切片语法 $a[:, j]$ 对齐。]。那么多元线性拟合时的残差向量变为了
#footnote[记号说明：在公式中我特意将常数项放到了最前面，这是为了让它和多项式拟合的形式保持一致。]
$
  bold(upright(r)) = bold(upright(y)) - hat(bold(upright(y))) = bold(upright(y)) - (b bold(upright(x))^0 + w_1 bold(upright(x_(:1))) + w_2 bold(upright(x_(:2))) + dots + w_d bold(upright(x_(:d))))
$

如果回顾一下我们在多项式拟合一节的内容，就会发现这和多项式时的残差向量

$ bold(upright(r)) = bold(upright(y)) - (a_0 bold(upright(x))^0 + a_1 bold(upright(x))^1 + dots + a_m bold(upright(x))^m) $

有着惊人的相似之处。细心的读者可能已经发现，如果令这些分量 $bold(upright(x))_(:1), bold(upright(x))_(:2), \ldots, bold(upright(x))_(:n)$ 分别为 $bold(upright(x))$ 的幂次组成的向量 $bold(upright(x))^1, bold(upright(x))^2, dots, bold(upright(x))^d$，那么我们得到的完完全全就是多项式拟合。这也就意味着，多项式拟合实际上可以视为多元线性拟合的一种特殊情况。

事已至此，我们似乎已经许多次遇到了这样一种情况：从一面看过去，是代数上，一组样本点上的线性拟合。但是从另一面看过去，确是在几何上找到高维空间的超平面中最接近给定点的向量。这里其实有不少精妙的数学原理
#footnote[写给数学基础好的读者：这本质上体现了代数与几何的#textOverSet("对偶性", "Duality")。]
，但是考虑到这里的主题是机器学习，我将只带读者简要地复习（或者学习）一下线性代数，更为系统性地从几种略有差距的视角
#footnote[
  几种视角：
  #textOverSet("整体", "Overall")解读、
  #textOverSet("按行", "Row-wise")解读、
  #textOverSet("按列", "Column-wise")解读、
  #textOverSet("按元素", "Element-wise")解读。
]
体会矩阵的本质。


在绘图讲解前，我首先要感谢
#link("https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra", "《线性代数的艺术》")
（#emph[The Art of Linear Algebra]）

这篇笔记，我第一次读到便感到文中的插图绘制非常精妙。它的思路是顺着 Gilbert Strang 教授书籍
#link("https://math.mit.edu/~gs/everyone/", "《写给所有人的线性代数》") 的思路，使用图形化的方式来解释线性代数的概念。认为可以看成是一本矩阵图鉴，对理解矩阵运算有着极大的帮助。

#link("https://www.bilibili.com/video/BV1ys411472E", "3Blue1Brown 的线性代数系列")
也是优质线性代数学习资源。这个制作精良的合集仅用不到两个小时的视频就清晰地从几何的角度讲明白了线性代数的基础知识，也是我入门线性代数的第一课。

矩阵有很多种#textOverSet("解读", "Interpretation")，不过我觉得大致可以按照是否把行看作一个整体以及是否把列看作一个整体来分为四类。