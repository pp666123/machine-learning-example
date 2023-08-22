const tf = require("@tensorflow/tfjs-node");

// 假设我们有一些中文句子作为训练数据
const chineseSentences = [
  "你好吗？",
  "我喜欢学习深度学习。",
  // ...
];

// 假设对应的英文翻译
const englishTranslations = [
  "How are you?",
  "I enjoy learning deep learning.",
  // ...
];

// 创建一个简单的序列到序列模型
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: chineseSentences.length, outputDim: 16, inputLength: 5 }));
model.add(tf.layers.lstm({ units: 16, returnSequences: true }));
model.add(tf.layers.lstm({ units: 16, returnSequences: true }));
model.add(
  tf.layers.timeDistributed({ layer: tf.layers.dense({ units: englishTranslations.length, activation: "softmax" }) })
);

// 编译模型
model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });

// 准备训练数据
const chineseSentencesTensor = tf.tensor2d(
  chineseSentences.map((str) => stringToCharIndices(str)),
  [chineseSentences.length, 5]
);

const englishTranslationsTensor = tf.tensor3d(
  englishTranslations.map((str) => stringToCharIndicesArray(str)),
  [englishTranslations.length, englishTranslations[0].length, 1]
);

// 将字符串转换为字符索引数组
function stringToCharIndices(str) {
  return Array.from(str).map((char) => char.charCodeAt(0));
}

function stringToCharIndicesArray(str) {
  return Array.from(str).map((char) => [char.charCodeAt(0)]);
}

// 进行训练
model
  .fit(chineseSentencesTensor, englishTranslationsTensor, { epochs: 50 })
  .then((info) => {
    console.log("训练完成");
  })
  .catch((error) => {
    console.error("训练出错:", error);
  });
