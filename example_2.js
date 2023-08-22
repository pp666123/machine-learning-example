const tf = require("@tensorflow/tfjs-node");

// 定义一个简单的线性回归模型
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// 准备训练数据
const x = tf.tensor([1, 2, 3, 4], [4, 1]);
const y = tf.tensor([2, 4, 6, 8], [4, 1]);

// 进行训练
model
  .fit(x, y, { epochs: 10 })
  .then((info) => {
    console.log("训练完成");

    // 使用训练好的模型进行预测
    const newX = tf.tensor([5], [1, 1]); // 输入值为 5
    const predictedY = model.predict(newX);
    console.log("预测结果:", predictedY.dataSync());
  })
  .catch((error) => {
    console.error("训练出错:", error);
  });
