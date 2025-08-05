#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot

// 空栈图
#let empty_stack() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底(栈顶)标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底(栈顶)]], anchor: "north")
    
    // 读写头
    line((0.5, 1.75), (0.5, 1.25), mark: (end: ">"))
    content((0.5, 1.9), [读写头], anchor: "south")
    
    // 标题
    content((4.5, -0.8), [空栈], anchor: "center")
  })
}

// 大小为1的栈图
#let stack_size_1() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 填充的部分
    rect((0, 0), (1, 1), fill: green.transparentize(50%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底]], anchor: "north")
    
    // 栈顶标记
    line((1, 0), (1, 1), stroke: red + 2pt)
    content((1, -0.3), [#text(red)[栈顶]], anchor: "north")
    
    // 读写头
    line((1.5, 1.75), (1.5, 1.25), mark: (end: ">"))
    content((1.5, 1.9), [读写头], anchor: "south")
    
    // 栈中的数字
    content((0.5, 0.5), [$1$], anchor: "center")
    
    // 标题
    content((4.5, -0.8), [栈 (大小为 1)], anchor: "center")
  })
}

// 大小为2的栈图
#let stack_size_2() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 填充的部分
    rect((0, 0), (2, 1), fill: green.transparentize(50%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底]], anchor: "north")
    
    // 栈顶标记
    line((2, 0), (2, 1), stroke: red + 2pt)
    content((2, -0.3), [#text(red)[栈顶]], anchor: "north")
    
    // 读写头
    line((2.5, 1.75), (2.5, 1.25), mark: (end: ">"))
    content((2.5, 1.9), [读写头], anchor: "south")
    
    // 栈中的数字
    content((0.5, 0.5), [$1$], anchor: "center")
    content((1.5, 0.5), [$2$], anchor: "center")
    
    // 标题
    content((4.5, -0.8), [栈 (大小为 2)], anchor: "center")
  })
}

// 弹栈图
#let stack_pop() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 填充的部分
    rect((0, 0), (1, 1), fill: green.transparentize(50%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底]], anchor: "north")
    
    // 栈顶标记
    line((1, 0), (1, 1), stroke: red + 2pt)
    content((1, -0.3), [#text(red)[栈顶]], anchor: "north")
    
    // 读写头
    line((1.5, 1.75), (1.5, 1.25), mark: (end: ">"))
    content((1.5, 1.9), [读写头], anchor: "south")
    
    // 栈中的数字
    content((0.5, 0.5), [$1$], anchor: "center")
    
    // 弹出箭头
    line((1.5, -0.25), (1.5, -0.75), mark: (end: ">"))
    content((1.5, -1), [弹出 $2$], anchor: "north")
    
    // 标题
    content((4.5, -0.8), [栈 (大小为 1)], anchor: "center")
  })
}

// 函数调用栈图
#let function_call_stack() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 填充的部分
    rect((0, 0), (3, 1), fill: green.transparentize(50%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底]], anchor: "north")
    
    // 栈顶标记
    line((3, 0), (3, 1), stroke: red + 2pt)
    content((3, -0.3), [#text(red)[栈顶]], anchor: "north")
    
    // 读写头
    line((3.5, 1.75), (3.5, 1.25), mark: (end: ">"))
    content((3.5, 1.9), [读写头], anchor: "south")
    
    // 栈中的内容
    content((0.5, 0.5), [... ], anchor: "center")
    content((1.5, 0.5), [... ], anchor: "center")
    content((2.5, 0.7), [next], anchor: "center")
    content((2.5, 0.3), [inst.], anchor: "center")
    
    // 标题
    content((4.5, -1), [栈 (大小为 3)], anchor: "center")
  })
}

// 函数返回栈图
#let function_return_stack() = {
  canvas(length: 3em, {
    import draw: *
    
    // 背景矩形
    rect((-1, 0), (10, 1), fill: gray.transparentize(85%), stroke: none)
    
    // 填充的部分
    rect((0, 0), (2, 1), fill: green.transparentize(50%), stroke: none)
    
    // 边框
    line((-1, 0), (10, 0))
    line((-1, 1), (10, 1))
    
    // 垂直分割线
    for x in range(0, 10) {
      line((x, 0), (x, 1))
    }
    
    // 栈底标记
    line((0, 0), (0, 1), stroke: red + 2pt)
    content((0, -0.3), [#text(red)[栈底]], anchor: "north")
    
    // 栈顶标记
    line((2, 0), (2, 1), stroke: red + 2pt)
    content((2, -0.3), [#text(red)[栈顶]], anchor: "north")
    
    // 读写头
    line((2.5, 1.75), (2.5, 1.25), mark: (end: ">"))
    content((2.5, 1.9), [读写头], anchor: "south")
    
    // 栈中的内容
    content((0.5, 0.5), [... ], anchor: "center")
    content((1.5, 0.5), [... ], anchor: "center")
    
    // 弹出箭头
    line((2.5, -0.25), (2.5, -0.75), mark: (end: ">"))
    content((2.5, -1), [弹出指令], anchor: "north")
    
    // 标题
    content((4.5, -0.6), [栈 (大小为 2)], anchor: "center")
  })
}

// 内存分配图
#let memory_layout() = {
  canvas(length: 3em, {
    import draw: *
    
    let width = 13
    let height = 0.2
    
    // 可自由使用内存
    rect((0, 0), (width, height), stroke: none, fill: gray)
    content((11.5, -0.3), [#text(gray)[可自由使用内存]], anchor: "north") 
    // r0-r7 寄存器
    rect((0, 0), (0.8, height), fill: red, stroke: none)
    content((0.0, -0.3), [#text(red)[`r0-r7`]], anchor: "north")
    
    // sp 栈指针
    rect((0.8, 0), (0.9, height), fill: orange, stroke: none)
    content((0.8, height + 0.1), [#text(orange)[`sp`]], anchor: "south")
    
    // ans 答案
    rect((0.9, 0), (1.0, height), fill: yellow, stroke: none)
    content((0.95, -0.3), [#text(yellow)[`ans`]], anchor: "north")
    
    // arg0-arg3 参数
    rect((1.0, 0), (1.4, height), fill: green, stroke: none)
    content((1.8, height + 0.1), [#text(green)[`arg0-arg3`]], anchor: "south")
    
    // 栈
    rect((2, 0), (10, height), fill: blue, stroke: none)
    content((6, -0.3), [#text(blue)[栈]], anchor: "north")
    
    // 边框
    line((0, 0), (width, 0))
    line((0, 0), (0, height))
    line((0, height), (width, height))
    
    // 栈底箭头
    line((2, -0.5), (2, -0.1), mark: (end: ">"), stroke: blue)
    content((2, -0.6), [#text(blue)[栈底]], anchor: "north")
    

  })
}

// 显示所有绘图函数
#empty_stack()
#v(1em)
#stack_size_1()
#v(1em)
#stack_size_2()
#v(1em)
#stack_pop()
#v(1em)
#function_call_stack()
#v(1em)
#function_return_stack()
#v(1em)
#memory_layout()