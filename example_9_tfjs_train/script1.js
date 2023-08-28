const image = new Image();
image.src = "1.jpg";

async function processImage(image) {
  const canvas = document.createElement("canvas");
  // 調整寬高為2的乘冪
  canvas.width = Math.pow(2, Math.ceil(Math.log2(image.width)));
  canvas.height = Math.pow(2, Math.ceil(Math.log2(image.height)));

  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  // 將 canvas 中的圖像轉換為張量
  const imageTensor = tf.browser.fromPixels(canvas);

  // 顯示在html
  const surface = tfvis.visor().surface({ name: "Input Data Examples", tab: "Input Data" });
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = "margin: 4px;";
  await tf.browser.toPixels(imageTensor, canvas);
  surface.drawArea.appendChild(canvas);

  imageTensor.dispose();
  console.log(imageTensor);

  // 建立模型
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  // 設定優化器
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  // 訓練
  // model.fit(xs, ys, {
  //   epochs: 500,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => {
  //       console.log("Epoch:" + epoch + " Loss:" + logs.loss);
  //     },
  //   },
  // });
}

// 圖像加載
image.onload = function () {
  processImage(image);
};
