#let template(doc) = [
  #set document(
    title: "understanding_nn_typst",
    author: "RUSRUSHB",
    date: auto,
  )

  #set heading(numbering: "1.1")

  #set par(justify: true, first-line-indent: (amount: 2em, all: true), leading: 1.2em)  // 段首缩进2em, 行间距1.2em
  // TODO: footnote会被换行，能修改吗
  // “一个经典例子是……”

  #set list(indent: 2em, tight: false)  // 列表缩进2em

  #set text(font: (
    (name: "Times New Roman", covers: "latin-in-cjk"), // 西文字体
    "NSimSun" // 中文字体

  ), lang: "zh", size: 12pt)  // 字体设置

  #show math.equation: set text(font: "Latin Modern Math")  // 公式字体设置

  #show figure: set block(inset: (top: 0.5em, bottom: 0.5em))  // figure上下间距

  #doc  // Apply the document settings when invoked `template`
]


#let overAnotate(mainText, overText) = box(
  align(center, 
    text((text(overText, size: 8pt)+"\n"+mainText)),
    )
  )
)  // 弃用了。换行符会影响行间距

#let textOverSet(mainText, overText) = [
  $limits(mainText)^overText$
]  //TODO: 字体？宋体不支持公式

#let recommend(title, text) = [
  
  #block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
  
  (title +"\n" + text),  //TODO: 
)
]