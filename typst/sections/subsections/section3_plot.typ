#import "@preview/cetz:0.4.0"

#let nn_elements = {

    figure(placement: none, caption: "神经网络的五个要素及其关系", gap: 1.5em,
    cetz.canvas({
        import cetz.draw: *;
        // 设置默认样式
        set-style(rect: (stroke: none))
        set-style(line: (mark: (end: "straight")))
        
        // box1
        rect((-0.5, -0.5), (0.5, 0.5), fill: aqua, name: "box1");
        content(("box1"), $x$)
        content((rel: (0, -0.75), to: "box1"), "输入");

        line((rel: (0.75, 0), to: "box1"), (rel: (3, 0), to: "box1"), name: "line1")
        content((rel: (0, 0.25), to: "line1"), "编码器")

        // box2
        rect((3.5, -0.5), (4.5, 0.5), fill: aqua, name: "box2");
        content(("box2"), $x'$)
        content((rel: (0, -0.75), to: "box2"), "向量输入");

        line((rel: (0.75, 0), to: "box2"), (rel: (3, 0), to: "box2"), name: "line2")
        content((rel: (0, 0.25), to: "line2"), "模型")

        // box3
        rect((7.5, -0.5), (8.5, 0.5), fill: lime, name: "box3");
        content(("box3"), $y'$)
        content((rel: (0, -0.75), to: "box3"), "向量输出");

        line((rel: (0.75, 0), to: "box3"), (rel: (3, 0), to: "box3"))

        // box4
        rect((11.5, -0.5), (12.5, 0.5), fill: silver, name: "box4");
        content(("box4"), $l$)
        content((rel: (0, -0.75), to: "box4"), "损失");

        // box5
        rect((7.5, -3.5), (8.5, -2.5), fill: lime, name: "box5");
        content(("box5"), $y$)
        content((rel: (0, -0.75), to: "box5"), "输出");
        line((rel: (0, -1), to: "box3"), (rel: (0, -2.25), to: "box3"), name: "line3")
        content((rel: (0.75, 0), to: "line3"), "解码器")

        // box6
        rect((7.5, 2), (8.5, 3), fill: yellow, name: "box6");
        content(("box6"), $o$)
        content((rel: (0, -0.75), to: "box6"), "优化器");

        line((rel: (0, 0.75), to: "box4"), (rel: (4, 0), to: "box6"), name: "line4", mark: none)
        line(("line4.end"), (rel: (0.75, 0), to: "box6"), name: "line5")
        content((rel: (0, 0.25), to: "line5"), "优化信息")

        line((rel: (-0.75, 0), to: "box6"), (rel: (0, 2.5), to: "line2"), name: "line6", mark: none)
        line(("line6.end"), (rel: (0, 0.75), to: "line2"), name: "line7")
        content((rel: (-1, 0), to: "line7"), "参数更新")

        // bigbox

        rect((2.5, -1.25), (13.5, 3.5), stroke: (dash: "dashed"))
      })
      )
}

#let relu_plot = {
  figure(placement: none, caption: "ReLU 函数图像", gap: 1.5em,
    cetz.canvas({
        import cetz.draw: *;

        line((-2, 0), (2, 0), name: "x-axis", mark: (end: "straight"));
        line((0, -1), (0, 2), name: "y-axis", mark: (end: "straight"));

        content((rel: (0, -.25), to:"x-axis.end") , $x$)
        content((rel: (-.25, 0), to:"y-axis.end") , $y$)

        line((-2, 0), (0, 0), stroke: (paint: blue, thickness: 2pt));
        line((0, 0), (2, 2), stroke: (paint: blue, thickness: 2pt));

      })
    )
}

#let nn_abs = {
  figure(placement: none, caption: $"神经网络表示" |x| "的结构"$, gap: 1.5em,
    cetz.canvas({
        import cetz.draw: *;

        set-style(circle: (radius: 0.55, fill: white));
        circle((-3, 0), name: "c1", );
        circle((0, 1.5), name: "c2");
        circle((0, -1.5), name: "c3");
        circle((3, 0), name: "c4");

        content("c1", $x$);
        content("c2", $x_1^((1))$);
        content("c3", $x_2^((1))$);
        content("c4", $|x|$);

        set-style(line: (mark: (end: "straight")));
        line(("c1"), ("c2"), stroke: (paint: red), name: "l1");
        line(("c1"), ("c3"), stroke: (paint: blue), name: "l2");
        line(("c2"), ("c4"), stroke: (paint: red), name: "l3");
        line(("c3"), ("c4"), stroke: (paint: red), name: "l4");

        content((rel: (-0.25, 0.25), to: "l1"), text(red)[$1$], fill: red, color: red);
        content((rel: (-0.25, -0.25), to: "l2"), text(blue)[$-1$], fill: blue, color: blue);
        content((rel: (0.25, 0.25), to: "l3"), text(red)[$1$], fill: red, color: red);
        content((rel: (0.25, -0.25), to: "l4"), text(red)[$1$], fill: red, color: red);

        content((rel: (0, 1), to: "c2"), text(silver)[$+0$]);
        content((rel: (0, 1), to: "c3"), text(silver)[$+0$]);
        content((rel: (0, 1), to: "c4"), text(silver)[$+0$]);

        content((rel: (0, -1), to: "c2"), "ReLU");
        content((rel: (0, -1), to: "c3"), "ReLU");

        rect((-1, -3), (1, 3), stroke: (dash: "dashed"), name: "box");
        content((rel: (0, 0.5), to: "box.north"), "中间层");

      })
    )
}

#nn_elements
#relu_plot
#nn_abs