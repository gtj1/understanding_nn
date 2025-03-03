#set heading(numbering: "1.1")
#set text(font: (
  (name: "Times New Roman", covers: "latin-in-cjk"), // 西文字体
  "NSimSun" // 中文字体
), lang: "zh")

// TODO: Formula font: latin mordern math

#set document(
title: "understanding_nn_typst",
author: "RUSRUSHB",
date: auto,
)


#align(center, text(17pt)[
  如何理解神经网络——信息量、压缩与智能
])
#outline()

= 从函数拟合开始

== 最简单的规律——简单线性回归

#figure(
  image("img\linear_regression.png", width: 50%),
  caption: ("简单线性回归\n图源："
  +underline(link("https://en.wikipedia.org/wiki/Linear_regression")[#text("Wikipedia", fill: blue)]))
  )

虽然 #box(align(center, (text("Linear Regression", size: 8pt)+"\n线性回归")))的名字叫做
