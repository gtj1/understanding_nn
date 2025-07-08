#let template(doc) = {
  // First set up all styles
  set document(
    title: "understanding_nn_typst",
    author: "RUSRUSHB",
    date: auto,
  )

  set heading(numbering: "1.1.1")

  set par(justify: true, first-line-indent: 2em, leading: 1.2em)  // 段首缩进2em, 行间距1.2em

  set text(font: (
    "Times New Roman",  // 西文字体
    "NSimSun" // 中文字体
  ), lang: "zh", size: 12pt)  // 字体设置

  show raw: set text(font: ("Consolas", "SimHei"))
  
  show raw.where(block: true): block.with(
    fill: luma(240),    // 背景色
    radius: 2pt,        // 圆角半径
    inset: 8pt,         // 内边距
    width: 100%,        // 宽度
    stroke: luma(200),  // 边框颜色
  )

  show link: set text(fill: blue)  // 所有链接设置为蓝色

  show figure: set block(inset: (top: 0.5em, bottom: 0.5em))  // figure上下间距

  set list(indent: 2em)

  set enum(indent: 2em)


  // Then return the document
  doc
}


#let textOverSet(mainText, overText) = [
  $limits(mainText)^overText$
]

#let recommend(Title, Text) = {

  let Title = text(font:("SimHei"), weight: "bold", size: 1.2em, fill: blue)[#Title]
  let Text = text(font:("Times New Roman", "SimSun"))[#Text]
  
  show raw: set text(font:("SimKai", "Times New Roman"), style: "italic", size: 10pt)
  show link: set text(size: 10pt)
  
  block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    width: 100%,

    
    par(first-line-indent: 0em, {
      Title
      v(1em)
      // line(length: 100%)  //TODO: 怎么分割？
      Text
    })
    
  )
}

#let v1 = v(1em)
#let h2 = h(2em)