const imageUpload = document.getElementById('image-upload');
const predictionElement = document.getElementById('prediction');

async function loadModel() {
    const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1', { fromTFHub: true });
    return model;
}

const modelPromise = loadModel();


function preprocessImage(image) {
    return tf.tidy(() => {
        const widthToHeight = image.shape[1] / image.shape[0];
        let squareCrop;
        if (widthToHeight > 1) {
            const heightToWidth = image.shape[0] / image.shape[1];
            const cropTop = (1 - heightToWidth) / 2;
            const cropBottom = 1 - cropTop;
            squareCrop = [[cropTop, 0, cropBottom, 1]];
        } else {
            const cropLeft = (1 - widthToHeight) / 2;
            const cropRight = 1 - cropLeft;
            squareCrop = [[0, cropLeft, 1, cropRight]];
        }
        const croppedImage = tf.image.cropAndResize(tf.expandDims(image), squareCrop, [0], [224, 224]);
        return croppedImage.div(255);
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
