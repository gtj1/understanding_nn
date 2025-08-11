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

// 图层蒙版应用效果示意图
#let mask_and() = {
  canvas(length: 1cm, {
    import draw: *

    // 将原图组移动到左侧
    translate((-3, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.55), (2.05, 1.45), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 图像内容
      content((0, 0), image("../../../img/img_no_mask.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[原图], anchor: "center")
    })
    
    // 重置变换
    translate((3, 0))
    
    // 中心乘号
    content((0, 0), text(14pt)[$times$], anchor: "center")
    
    // 右侧蒙版部分 (向右偏移3个单位)
    translate((3, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.55), (2.05, 1.45), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 蒙版内容
      content((0, 0), image("../../../img/layer_mask.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[蒙版], anchor: "center")
    })
    
    // 重置变换
    translate((-3, 0))
    
    // 向下箭头
    line((0, -2), (0, -3), stroke: (paint: black, thickness: 2pt), mark: (end: ">"))
    
    // 底部结果部分 (向下偏移5个单位)
    translate((0, -5))
    group({
      // 阴影效果
      rect((-1.95, -1.55), (2.05, 1.45), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 结果内容
      content((0, 0), image("../../../img/img_with_mask.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[蒙版应用], anchor: "center")
    })
  })
}

// 蒙版反转示意图
#let mask_not() = {
  canvas(length: 1cm, {
    import draw: *
    
    // 左侧原蒙版
    translate((-4, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.55), (2.05, 1.45), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 蒙版内容
      content((0, 0), image("../../../img/layer_mask.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[原蒙版], anchor: "center")
    })
    
    // 重置变换并移到中心
    translate((4, 0))
    
    // 中心箭头和Not标签
    line((-1.3, 0), (1.3, 0), stroke: (paint: black, thickness: 2pt), mark: (end: ">"))
    content((0, 0.3), text(10pt)[Not], anchor: "center")
    
    // 右侧反色蒙版
    translate((4, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.55), (2.05, 1.45), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 反色蒙版内容
      content((0, 0), image("../../../img/layer_mask_rev.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[反色蒙版], anchor: "center")
    })
  })
}

// 蒙版逻辑运算示意图
#let mask_operations() = {
  canvas(length: 1cm, {
    import draw: *
    
    // 外层容器背景（灰色阴影）
    rect((-4.45, -2.35), (4.55, 1.85), fill: gray.lighten(60%), stroke: none)
    rect((-4.5, -2.4), (4.5, 1.8), fill: white, stroke: (paint: gray, thickness: 2pt))
    
    // 顶部两个蒙版
    // 左侧蒙版1
    translate((-2.2, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.45), (2.05, 1.55), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 蒙版内容
      content((0, 0), image("../../../img/layer_mask.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[蒙版 1], anchor: "center")
    })
    
    // 重置变换
    translate((2.2, 0))
    
    // 右侧蒙版2
    translate((2.2, 0))
    group({
      // 阴影效果
      rect((-1.95, -1.45), (2.05, 1.55), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // 蒙版内容
      content((0, 0), image("../../../img/layer_mask_2.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[蒙版 2], anchor: "center")
    })
    
    // 重置变换
    translate((-2.2, 0))
    
    // 三个箭头和标签
    // And箭头（左）
    line((-2, -2.7), (-3.5, -4.2), stroke: (paint: black, thickness: 2pt), mark: (end: ">"))
    rect((-3.8, -3.6), (-3.2, -3.2), fill: white, stroke: none)
    content((-3.5, -3.4), text(10pt)[And], anchor: "center")
    
    // Or箭头（中）
    line((0, -2.7), (0, -4.2), stroke: (paint: black, thickness: 2pt), mark: (end: ">"))
    rect((-0.3, -3.6), (0.3, -3.2), fill: white, stroke: none)
    content((0, -3.4), text(10pt)[Or], anchor: "center")
    
    // Xor箭头（右）
    line((2, -2.7), (3.5, -4.2), stroke: (paint: black, thickness: 2pt), mark: (end: ">"))
    rect((3.2, -3.6), (3.8, -3.2), fill: white, stroke: none)
    content((3.5, -3.4), text(10pt)[Xor], anchor: "center")
    
    // 底部三个结果图像
    // And结果（左下）
    translate((-5, -6))
    group({
      // 阴影效果
      rect((-1.95, -1.45), (2.05, 1.55), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // And结果内容
      content((0, 0), image("../../../img/layer_mask_and.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[And 结果], anchor: "center")
    })
    
    // 重置变换
    translate((5, 0))
    
    // Or结果（中下）
    group({
      // 阴影效果
      rect((-1.95, -1.45), (2.05, 1.55), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // Or结果内容
      content((0, 0), image("../../../img/layer_mask_or.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[Or 结果], anchor: "center")
    })
    
    // 移到右下
    translate((5, 0))
    
    // Xor结果（右下）
    group({
      // 阴影效果
      rect((-1.95, -1.45), (2.05, 1.55), fill: rgb("#a855f7").lighten(60%), stroke: none)
      // 图像区域
      rect((-2, -1.5), (2, 1.5), fill: white, stroke: (paint: rgb("#a855f7"), thickness: 2pt))
      // Xor结果内容
      content((0, 0), image("../../../img/layer_mask_xor.png", width: 4cm), anchor: "center")
      // 标签
      content((0, -1.8), text(10pt)[Xor 结果], anchor: "center")
    })
  })
}

// 独热编码可视化表格
#let onehot_encoding_visualization() = {
  canvas(length: 0.8cm, {
    import draw: *
    
    // 设置网格大小
    let cols = 4
    let rows = 10
    let cell_size = 0.6
    
    // 绘制背景
    rect((0, 0), (cols * cell_size, rows * cell_size), 
         fill: gray.lighten(85%), stroke: none)
    
    // 定义独热编码数据 (对应标签序列: 3,0,2,0,2,3,0,2,0,1)
    let onehot_data = (
      (0, 0, 0, 1), // 标签3
      (1, 0, 0, 0), // 标签0
      (0, 0, 1, 0), // 标签2
      (1, 0, 0, 0), // 标签0
      (0, 0, 1, 0), // 标签2
      (0, 0, 0, 1), // 标签3
      (1, 0, 0, 0), // 标签0
      (0, 0, 1, 0), // 标签2
      (1, 0, 0, 0), // 标签0
      (0, 1, 0, 0), // 标签1
    )
    
    // 绘制红色单元格（值为1的位置）
    for (row, data) in onehot_data.enumerate() {
      for (col, value) in data.enumerate() {
        if value == 1 {
          rect((col * cell_size, (rows - 1 - row) * cell_size), 
               ((col + 1) * cell_size, (rows - row) * cell_size),
               fill: red.transparentize(50%), stroke: none)
        }
      }
    }
    
    // 绘制网格线
    for x in range(cols + 1) {
      line((x * cell_size, 0), (x * cell_size, rows * cell_size), 
           stroke: (paint: gray.lighten(30%), thickness: 0.5pt))
    }
    for y in range(rows + 1) {
      line((0, y * cell_size), (cols * cell_size, y * cell_size), 
           stroke: (paint: gray.lighten(30%), thickness: 0.5pt))
    }
    
    // 添加列标题（类别编号）
    content((2 * cell_size, rows * cell_size + 0.6), text(10pt)[类别编号], anchor: "center")
    for col in range(cols) {
      content(((col + 0.5) * cell_size, rows * cell_size + 0.2), 
              text(9pt)[#str(col)], anchor: "center")
    }
    
    // 添加底部省略号
    content((2 * cell_size, -0.3), text(10pt)[......], anchor: "center")
  })
}


#onehot_encoding_visualization()

