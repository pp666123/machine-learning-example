// 定義變量
// Initialize the Image Classifier method with MobileNet. A callback needs to be passed.
let classifier;

// A variable to hold the image we want to classify
let img;

// 載入圖片&訓練模型
function preload() {
  classifier = ml5.imageClassifier("MobileNet");
  img = loadImage("images/bird.png");
}

// p5.js顯示圖片ˋ
function setup() {
  createCanvas(400, 400);
  classifier.classify(img, gotResult);
  // https://p5js.org/reference/#/p5/image
  image(img, 0, 0, 400, 400);
  // 繪製另外一張聚焦的圖片
  image(img, 0, 0, 80, 80, 180, 120, 200, 200);
}

// A function to run when we get any errors and the results
function gotResult(error, results) {
  // Display error in the console
  if (error) {
    console.error(error);
  } else {
    // The results are in an array ordered by confidence.
    console.log(results);
    createDiv(`Label: ${results[0].label}`);
    createDiv(`Confidence: ${nf(results[0].confidence, 0, 2)}`);
  }
}

function mousePressed() {
  img.resize(500, 100);
}
