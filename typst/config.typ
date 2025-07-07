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

  show link: set text(fill: blue)  // 所有链接设置为蓝色

  show figure: set block(inset: (top: 0.5em, bottom: 0.5em))  // figure上下间距

  // Then return the document
  doc
}


#let textOverSet(mainText, overText) = [
  $limits(mainText)^overText$
]

#let recommend(Title, Text) = {

  let Title = text(font:("Times New Roman", "KaiTi"), weight: "bold", size: 1.2em, fill: blue)[#Title]
  let Text = text(font:("Times New Roman", "KaiTi"))[#Text]
  
  block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    width: 100%,
    
    {
      Title
      line(length: 100%)
      h(2em)
      Text
      // str(text, base: 10)
    }
  )
}