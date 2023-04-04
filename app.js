// import * as tf from '@tensorflow/tfjs';

const imageUpload = document.getElementById('image-upload');
const predictionElement = document.getElementById('prediction');

async function loadModel() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    return model;
}

const modelPromise = loadModel();

function preprocessImage(image) {
    // 画像をモデルが受け付ける形式にリサイズ・正規化する
    // この例では、MobileNetが受け付ける224x224サイズにリサイズ
    return tf.tidy(() => {
        const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
        const normalizedImage = resizedImage.div(255.0);
        const batchedImage = normalizedImage.expandDims(0);
        return batchedImage;
    });
}

async function classifyImage(image) {
    const model = await modelPromise;
    const preprocessedImage = preprocessImage(image);
    const prediction = model.predict(preprocessedImage);
    return prediction;
}

async function fileToImageData(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        const img = new Image();

        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            resolve(imageData);
        };

        img.onerror = (err) => {
            reject(err);
        };

        reader.onload = (event) => {
            img.src = event.target.result;
        };

        reader.onerror = (err) => {
            reject(err);
        };

        reader.readAsDataURL(file);
    });
}

async function fetchClassLabels() {
    const response = await fetch('imagenet_class_index.json');
    const classIndex = await response.json();
    const classLabels = Object.values(classIndex).map(entry => entry[1]);
    return classLabels;
}

const classLabelsPromise = fetchClassLabels();

async function indexToLabel(index) {
    const classLabels = await classLabelsPromise;
    return classLabels[index];
}


imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const imageData = await fileToImageData(file);
    const image = await tf.browser.fromPixelsAsync(imageData);
    const prediction = await classifyImage(image);

    // 選択した画像を表示する
    const imagePreview = document.getElementById('image-preview');
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.style.display = 'block';

    // 結果を表示する
    // この例では、最も確率の高いクラスのインデックスを表示
    const maxConfidenceIndex = prediction.argMax(1).dataSync()[0];
    const label = await indexToLabel(maxConfidenceIndex);
    predictionElement.textContent = `分類結果: (${label})`;
});
