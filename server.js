import express from "express";
import cors from "cors";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const BATCH_SIZE = 4;
const EPOCH = 100;
const LEARNING_RATE = 0.001
const __dirname = path.dirname(fileURLToPath(import.meta.url));
let modelType

const app = express();
const port = 8081;
let model, baseModel, customModel, combinedModel, localModel;
let classNames = [],
    traningDataInputs = [],
    trainingDataOutputs = [];
let result, confidence;

// function
//load model
async function loadMobileNetFeatureModel() {
    const url =
        "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
    model = await tf.loadLayersModel(url);
    console.log("MobileNet v2 loaded successfully!");

    const layer = model.getLayer("global_average_pooling2d_1");
    baseModel = tf.model({ inputs: model.inputs, outputs: layer.output });
    // baseModel.summary()

    tf.tidy(function () {
        let answer = baseModel.predict(
            tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
        );
        console.log(answer.shape);
    });
}

async function loadLocalModel() {
    try {
        localModel = await tf.loadLayersModel("file://./models/model.json");
        localModel.summary()
        console.log("Local Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load Local model:", error);
    }
}

//upload data
function upload() {
    //convert base64 to image
    //save to database
}

//precessor (base64 to )
function preprocess(image) {
    return tf.tidy(() => {
        // Ensure that 'image' is a string
        if (typeof image !== "string") {
            throw new TypeError(
                "Expected image to be a base64 string, but received " + typeof image
            );
        }

        let base64Image = image.replace(/^data:image\/(png|jpeg);base64,/, "");
        const buffer = Buffer.from(base64Image, "base64");
        let imageAsTensor = tf.node.decodeImage(buffer, 3);
        let resizedTensorImage = tf.image.resizeBilinear(
            imageAsTensor,
            [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
            true
        );
        let normalizedTensorImage = resizedTensorImage.div(255);
        return normalizedTensorImage.expandDims();
    });
}

//extract features
function extractFeature(image) {
    let processedImage = preprocess(image);
    return tf.tidy(() => {
        let imageFeatures = baseModel.predict(processedImage);
        return imageFeatures.squeeze();
    });
}

//train
async function train() {
    const imageDir = path.join(__dirname, "images");
    const classFolders = await fs.promises.readdir(imageDir, {
        withFileTypes: true,
    });

    for (let folder of classFolders) {
        if (folder.isDirectory()) {
            let folderPath = path.join(imageDir, folder.name);
            let images = await fs.promises.readdir(folderPath);

            for (let imageFile of images) {
                let imagePath = path.join(folderPath, imageFile);
                let imageBuffer = fs.readFileSync(imagePath);
                let imageBase64 = `data:image/jpeg;base64,${imageBuffer.toString(
                    "base64"
                )}`;
                let features = extractFeature(imageBase64); // Ensure to await here

                traningDataInputs.push(features);
                console.log("input: " + features);
                trainingDataOutputs.push(classNames.indexOf(folder.name));
                console.log("output: " + classNames.indexOf(folder.name));
            }
        }
    }

    // Stack inputs, convert outputs to tensor, and perform training as before
    tf.util.shuffleCombo(traningDataInputs, trainingDataOutputs);
    let inputsAsTensor = tf.stack(traningDataInputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    let oneHotOutputs = tf.oneHot(outputsAsTensor, classNames.length);

    let results = await customModel.fit(inputsAsTensor, oneHotOutputs, {
        batchSize: BATCH_SIZE,
        validationSplit: 0.1,
        epochs: EPOCH,
        callbacks: [
            {
                onEpochEnd: (epoch, logs) =>
                    console.log("Data for epoch " + epoch, logs),
            },
            //   tf.callbacks.earlyStopping({
            //     monitor: 'val_loss',
            //     patience: 10  // Stops training if val_loss does not improve for 10 consecutive epochs
            // })
        ],
    });

    inputsAsTensor.dispose();
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();

    combinedModel = tf.sequential();
    combinedModel.add(baseModel);
    combinedModel.add(customModel);
    combinedModel.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
    });
    //   combinedModel.summary();
    await combinedModel.save("file://./models");
}

//predict
function predict(image) {
    if (localModel) {
        tf.tidy(function () {
            modelType = 'local'
            let prediction = localModel.predict(preprocess(image)).squeeze();
            let highestIndex = prediction.argMax().arraySync();
            let predictionArray = prediction.arraySync();
            result = classNames[highestIndex];
            confidence = Math.floor(predictionArray[highestIndex] * 100);
        });
    } else {
        tf.tidy(function () {
            modelType = 'new'
            let imageFeatures = baseModel.predict(preprocess(image));
            let prediction = customModel.predict(imageFeatures).squeeze();
            let highestIndex = prediction.argMax().arraySync();
            let predictionArray = prediction.arraySync();
            result = classNames[highestIndex];
            confidence = Math.floor(predictionArray[highestIndex] * 100);
        });
    }
}

//util
async function loadClassNames() {
    try {
        const filePath = path.join(__dirname, "classNames.json");
        // Use fs.promises.readFile to correctly handle the promise
        const jsonData = await fs.promises.readFile(filePath, "utf8");
        classNames = JSON.parse(jsonData);
        console.log(classNames); // Outputs: ['pro1', 'pro2']
        return classNames;
    } catch (error) {
        console.error("Failed to read or parse JSON file:", error);
    }
}

// server
app.use(express.json({ limit: "50mb" }));
app.use(cors());
await loadClassNames();
await loadMobileNetFeatureModel();
await loadLocalModel();
// await loadRasnetV2Model();

//add custom layer
customModel = tf.sequential();
customModel.add(
    tf.layers.dense({ inputShape: [1280], units: 128, activation: "relu" })
);
customModel.add(
    tf.layers.dense({ units: classNames.length, activation: "softmax" })
);
const optimizer = tf.train.adam(LEARNING_RATE);
customModel.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ["accuracy"],
});
customModel.summary()
//api
// app.post('/api/uploadData', async (req, res, next) => {
//     try {
//         const { className, images } = req.body;
//         const classPath = path.join(__dirname, 'images', className);

//         // Check if the directory exists, create if not
//         if (!fs.existsSync(classPath)) {
//             fs.mkdirSync(classPath, { recursive: true });
//         }

//         // Iterate over the base64 images array and convert them to image files
//         images.forEach((base64, index) => {
//             // Remove header from base64 string
//             const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');
//             const imagePath = path.join(classPath, `${className}_${index}.png`);
//             // Write file
//             fs.writeFileSync(imagePath, base64Data, { encoding: 'base64' });
//         });

//         res.status(200).json("Upload successfully");
//     } catch (error) {
//         console.error(error);
//         next(error);
//     }
// });

app.post("/api/train", async (req, res, next) => {
    try {
        await train();
        res.status(200).json("train successful");
    } catch (error) {
        next(error);
    }
});

app.post("/api/predict", async (req, res, next) => {
    try {
        const { model, image } = req.body;
        predict(image);
        res.status(200).json({ result, confidence, modelType });
    } catch (error) {
        next(error);
    }
});

//run server
app.listen(port);
