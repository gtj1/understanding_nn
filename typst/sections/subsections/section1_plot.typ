#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot

#let data_points = (
  (0.0, 0.423), (0.1, -0.726), (0.2, 1.96), (0.3, 0.065), (0.4, -0.355), 
  (0.5, 0.45), (0.6, -0.778), (0.7, 0.123), (0.8, 0.531), (0.9, -1.097), 
  (1.0, 1.218), (1.1, -0.018), (1.2, 0.363), (1.3, -2.308), (1.4, -0.688), 
  (1.5, 0.5), (1.6, 1.28), (1.7, -0.052), (1.8, 0.45), (1.9, -0.398), 
  (2.0, 0.207), (2.1, 0.237), (2.2, -1.486), (2.3, 0.244), (2.4, -0.215), 
  (2.5, -0.849), (2.6, 0.395), (2.7, -0.185), (2.8, -0.415), (2.9, -0.477), 
  (3.0, 1.11), (3.1, -0.435), (3.2, 0.833), (3.3, 1.451), (3.4, 0.147), 
  (3.5, 0.139), (3.6, -1.163), (3.7, 3.425), (3.8, -0.117), (3.9, 2.428), 
  (4.0, 1.964), (4.1, 1.613), (4.2, 0.992), (4.3, 0.949), (4.4, 1.707), 
  (4.5, 2.736), (4.6, 0.905), (4.7, 1.117), (4.8, 3.352), (4.9, 3.162), 
  (5.0, 2.226), (5.1, 3.536), (5.2, 1.403), (5.3, 2.448), (5.4, 2.652), 
  (5.5, 3.959), (5.6, 3.705), (5.7, 3.537), (5.8, 3.656), (5.9, 4.784), 
  (6.0, 3.247), (6.1, 4.589), (6.2, 2.857), (6.3, 4.194), (6.4, 4.053), 
  (6.5, 2.536), (6.6, 5.568), (6.7, 5.73), (6.8, 3.667), (6.9, 6.415), 
  (7.0, 4.927), (7.1, 5.545), (7.2, 6.754), (7.3, 6.447), (7.4, 7.352), 
  (7.5, 7.55), (7.6, 8.649), (7.7, 7.145), (7.8, 7.837), (7.9, 7.964), 
  (8.0, 9.074), (8.1, 8.135), (8.2, 10.401), (8.3, 8.005), (8.4, 10.431), 
  (8.5, 10.629), (8.6, 8.624), (8.7, 10.726), (8.8, 13.097), (8.9, 12.136), 
  (9.0, 12.849), (9.1, 10.127), (9.2, 14.107), (9.3, 13.364), (9.4, 14.538), 
  (9.5, 12.474), (9.6, 15.591), (9.7, 13.714), (9.8, 16.952), (9.9, 16.317), 
  (10.0, 16.764)
)

#let fo(x) = 0.25 * calc.pow(x, 2) - x + 1  // f_original
#let f1(x) = 1.5142676761793825 * x -3.3366750145602806
#let f2(x) = 0.24383409557403493 * calc.pow(x, 2) -0.9240732795609664 * x + 0.6865875624112963
#let f3(x) = 0.012628009463763832 * calc.pow(x, 3) + 0.054413953617577414 * calc.pow(x, 2) -0.17015585855533788 * x + 0.07400282332411103
#let f10(x) = -0.000004129005667 * calc.pow(x, 10) + 0.000200033877258 * calc.pow(x, 9) - 0.004061827595427 * calc.pow(x, 8) + 0.044810202155712 * calc.pow(x, 7) - 0.291097682876070 * calc.pow(x, 6) + 1.129425113256322 * calc.pow(x, 5) - 2.542208192992861 * calc.pow(x, 4) + 3.091776048493755 * calc.pow(x, 3) - 1.584353162512058 * calc.pow(x, 2) - 0.267618886184698 * x + 0.362912675589959

#let data_and_noise = {

  figure(placement: none, 
    canvas({
    import draw: *

    // Set-up a thin axis style

    plot.plot(size: (12, 8),
      x-tick-step: 2,
      // x-format: plot.formats.multiple-of,
      y-tick-step: 5, y-min: -2.5, y-max: 15,
      legend: "inner-north",
      axis-style: "school-book",
      x-label: $ x $,
      y-label: $ y $,
      {
        let domain = (0, 10)

        plot.add(data_points, label: "带噪声数据",
          style: (stroke: none),  // 去除折线
          mark: "o",
          mark-style: (stroke: blue, fill: blue),
        )
        plot.add(fo, domain: domain, label: "真实曲线",
          style: (stroke: red))
      })
  })
  )
}

#let fitting_3 = {
  figure(placement: none, 
    canvas({
      import draw: *

      plot.plot(size: (12, 8),
        x-tick-step: 2,
        // x-format: plot.formats.multiple-of,
        y-tick-step: 5, y-min: -2.5, y-max: 15,
        legend: "inner-north",
        axis-style: "school-book",
        x-label: $ x $,
        y-label: $ y $,
        {
          let domain = (0, 10)

          plot.add(data_points, label: "带噪声数据",
            style: (stroke: none),  // 去除折线
            mark: "o",
            mark-style: (stroke: blue, fill: blue),
          )
          plot.add(fo, domain: domain, label: "真实曲线",
            style: (stroke: red))
          plot.add(f1, domain: domain, label: "线性拟合",
            style: (stroke: orange))
          plot.add(f2, domain: domain, label: "二次拟合",
            style: (stroke: yellow))
          plot.add(f3, domain: domain, label: "三次拟合",
            style: (stroke: green))

        })
    })
  )
}

#let fitting_10 = {
  figure(placement: none, 
    canvas({
      import draw: *

      plot.plot(size: (12, 8),
        x-tick-step: 2,
        // x-format: plot.formats.multiple-of,
        y-tick-step: 5, y-min: -2.5, y-max: 15,
        legend: "inner-north",
        axis-style: "school-book",
        x-label: $ x $,
        y-label: $ y $,
        {
          let domain = (0, 10)

          plot.add(data_points, label: "带噪声数据",
            style: (stroke: none),  // 去除折线
            mark: "o",
            mark-style: (stroke: blue, fill: blue),
          )
          plot.add(fo, domain: domain, label: "真实曲线",
            style: (stroke: red))
          plot.add(f10, domain: domain, label: "十次拟合",
            style: (stroke: orange))
        })
    })
  )
}

#let fitting_10_large = {
  figure(placement: none, 
    canvas({
      import draw: *

      plot.plot(size: (12, 8),
        x-tick-step: 2,
        // x-format: plot.formats.multiple-of,
        y-tick-step: 20, y-min: -40, y-max: 20,
        legend: "inner-north",
        axis-style: "school-book",
        x-label: $ x $,
        y-label: $ y $,
        {
          let domain = (-2, 12)

          plot.add(data_points, label: "带噪声数据",
            style: (stroke: none),  // 去除折线
            mark: "o",
            mark-style: (stroke: blue, fill: blue),
          )
          plot.add(fo, domain: domain, label: "真实曲线",
            style: (stroke: red))
          plot.add(f10, domain: domain, label: "十次拟合",
            style: (stroke: orange),
            samples: 100  // 增加采样点数以获得更平滑的曲线
            )
        })
    })
  )
}

#let fo_n(x) = 0.25 * calc.pow((x+1)*5, 2) - (x+1)*5 + 1
#let f10_n(x) = 0.056914714852145956 * calc.pow(x, 10) -0.16883868373880576 * calc.pow(x, 9) +0.2748227691835253 * calc.pow(x, 8) +0.02818103309527224 * calc.pow(x, 7) +0.8506420254328089 * calc.pow(x, 6) +0.5250028027489858 * calc.pow(x, 5) +1.8635063516688952 * calc.pow(x, 4) +1.8797071938317609 * calc.pow(x, 3) +3.414068726324771 * calc.pow(x, 2) +6.0490130644654645 * x +2.5048936484111635

#let data_points_n = (
  (-1.0, 0.423), (-0.98, -0.726), (-0.96, 1.96), (-0.94, 0.065), (-0.92, -0.355), (-0.9, 0.45), (-0.88, -0.778), (-0.86, 0.123), (-0.84, 0.531), (-0.82, -1.097), (-0.8, 1.218), (-0.78, -0.018), (-0.76, 0.363), (-0.74, -2.308), (-0.72, -0.688), (-0.7, 0.5), (-0.68, 1.28), (-0.66, -0.052), (-0.64, 0.45), (-0.62, -0.398), (-0.6, 0.207), (-0.58, 0.237), (-0.56, -1.486), (-0.54, 0.244), (-0.52, -0.215), (-0.5, -0.849), (-0.48, 0.395), (-0.46, -0.185), (-0.44, -0.415), (-0.42, -0.477), (-0.4, 1.11), (-0.38, -0.435), (-0.36, 0.833), (-0.34, 1.451), (-0.32, 0.147), (-0.3, 0.139), (-0.28, -1.163), (-0.26, 3.425), (-0.24, -0.117), (-0.22, 2.428), (-0.2, 1.964), (-0.18, 1.613), (-0.16, 0.992), (-0.14, 0.949), (-0.12, 1.707), (-0.1, 2.736), (-0.08, 0.905), (-0.06, 1.117), (-0.04, 3.352), (-0.02, 3.162), (0.0, 2.226), (0.02, 3.536), (0.04, 1.403), (0.06, 2.448), (0.08, 2.652), (0.1, 3.959), (0.12, 3.705), (0.14, 3.537), (0.16, 3.656), (0.18, 4.784), (0.2, 3.247), (0.22, 4.589), (0.24, 2.857), (0.26, 4.194), (0.28, 4.053), (0.3, 2.536), (0.32, 5.568), (0.34, 5.73), (0.36, 3.667), (0.38, 6.415), (0.4, 4.927), (0.42, 5.545), (0.44, 6.754), (0.46, 6.447), (0.48, 7.352), (0.5, 7.55), (0.52, 8.649), (0.54, 7.145), (0.56, 7.837), (0.58, 7.964), (0.6, 9.074), (0.62, 8.135), (0.64, 10.401), (0.66, 8.005), (0.68, 10.431), (0.7, 10.629), (0.72, 8.624), (0.74, 10.726), (0.76, 13.097), (0.78, 12.136), (0.8, 12.849), (0.82, 10.127), (0.84, 14.107), (0.86, 13.364), (0.88, 14.538), (0.9, 12.474), (0.92, 15.591), (0.94, 13.714), (0.96, 16.952), (0.98, 16.317), (1.0, 16.764)
)

#let fitting_10_normalized = {
  figure(placement: none, 
    canvas({
      import draw: *

      plot.plot(size: (12, 8),
        x-tick-step: 0.5,
        // x-format: plot.formats.multiple-of,
        y-tick-step: 10, y-min: -5, y-max: 50,
        legend: "inner-north-west",
        axis-style: "school-book",
        x-label: $ x $,
        y-label: $ y $,
        {
          let domain = (-1.5, 1.5)

          plot.add(data_points_n, label: "带噪声数据",
            style: (stroke: none),  // 去除折线
            mark: "o",
            mark-style: (stroke: blue, fill: blue),
          )
          plot.add(fo_n, domain: domain, label: "真实曲线",
            style: (stroke: red))
          plot.add(f10_n, domain: domain, label: "十次拟合",
            style: (stroke: orange),
            samples: 100  // 增加采样点数以获得更平滑的曲线
            )
        })
    })
  )
}


// #data_and_noise
// #fitting_3
// #fitting_10
// #fitting_10_large
#fitting_10_normalized