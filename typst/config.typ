#let template(doc) = {
  // First set up all styles
  set document(
    title: "understanding_nn_typst",
    author: "RUSRUSHB",
    date: auto,
  )

  // 标题：编号
  set heading(numbering: "1.1.1")

  // 段落：两端对齐，段首缩进2em，行间距1.2em
  set par(justify: true, first-line-indent: 2em, leading: 1.2em)  // 段首缩进2em, 行间距1.2em

  // 正文：Times New Roman，NSimSun
  set text(font: (
    "Times New Roman",  // 西文字体
    "NSimSun" // 中文字体
  ), lang: "zh", size: 12pt)  // 字体设置

  show strong: set text(font: ("SimHei"), weight: "bold")

  // 代码：Consolas，SimHei
  show raw: set text(font: ("Consolas", "SimHei"))
  
  show raw.where(block: true): block.with(
    fill: luma(240),    // 背景色
    radius: 2pt,        // 圆角半径
    inset: 8pt,         // 内边距
    width: 100%,        // 宽度
    stroke: luma(200),  // 边框颜色
  )

  // 链接：下划线，蓝色
  show link: underline
  show link: set text(fill: blue) 

  // 引用：蓝色
  show ref: set text(fill: blue)

  // 脚注：红色
  show footnote: set text(fill: red)

  // 列表：缩进2em
  set list(indent: 2em)

  // 枚举：缩进2em
  set enum(indent: 2em)

  // 表格：居中
  show table: set align(center)

  // #show figure.where(kind: "table"): figure.with(supplement: [表])  // TODO:自动将表格的补充信息设置为表

  // Then return the document
  doc
}


#let textOverSet(mainText, overText) = [
  $limits(mainText)^overText$
]

#let recommend(Title, Text) = {

  let Title = text(font:("SimHei"), weight: "bold", size: 1.2em, fill: blue)[#Title]
  let Text = text(font:("Times New Roman", "SimSun"))[#Text]
  
  show raw: set text(font:("SimHei", "Times New Roman"), style: "italic", size: 10pt)
  show link: set text(size: 10pt)
  
  block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    width: 100%,
    stroke: luma(200),

    
    par(first-line-indent: 0em, {
      Title
      linebreak()
      Text
    })
    
  )
}

// 快速调整空间，用于排版
#let v1 = v(1em)
#let h2 = h(2em)