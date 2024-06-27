const express = require('express');
const bodyParser = require('body-parser');
const { Canvas, Image, ImageData } = require('canvas');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs-node');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());

const port = 3000;
const db = new sqlite3.Database('./database.sqlite');

// Configurar EJS como motor de plantillas
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render('index');
});

// Cargar los modelos
const loadModels = async () => {
    const modelPath = './models';
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
};

app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));

// Función para convertir imagen a tensor
const imageToTensor = (img) => {
    /*const canvas = new Canvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return tf.node.decodeImage(imageData, 3);*/

    /*const image = img.replace(
        /^data:image\/(png|jpeg);base64,/,
        ""
      );*/
      const b = Buffer.from(img, "base64");
      const tensor = tf.node.decodeImage(b, 3);
      return tensor;

};

// Endpoint para registrar un nuevo usuario
app.post('/register', async (req, res) => {
    try {
        const { id, name, ci, image } = req.body;
        const img = new Image();
        img.src = `data:image/jpeg;base64,${image}`;

        // Verificar que la imagen se haya cargado correctamente
        //img.onload = async () => {
        try {
            // Convertir la imagen a un tensor
            const tensor = imageToTensor(image);
            console.log('Tensor: ', tensor);
            const detections = await faceapi.detectSingleFace(tensor).withFaceLandmarks().withFaceDescriptor();
            if (!detections) {
                return res.status(400).json({ error: 'No face detected' });
            }

            const descriptor = JSON.stringify(Array.from(detections.descriptor));
            console.log('Descriptor: ', descriptor);
            db.run(`INSERT INTO users (id_cliente, name, ci, descriptor) VALUES (?, ?, ?, ?)`, [id, name, ci, descriptor], function (err) {
                if (err) {
                    return res.status(500).json({ error: err.message });
                }
                res.json({ id: this.lastID });
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
        //};

        img.onerror = (err) => {
            res.status(400).json({ error: 'Failed to load image' });
        };
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Endpoint para reconocer un usuario
app.post('/recognize', async (req, res) => {
    try {
        const { image } = req.body;
        const img = new Image();
        img.src = `data:image/jpeg;base64,${image}`;

        // Verificar que la imagen se haya cargado correctamente
        img.onload = async () => {
            try {
                // Convertir la imagen a un tensor
                const tensor = imageToTensor(img);

                const detections = await faceapi.detectSingleFace(tensor).withFaceLandmarks().withFaceDescriptor();
                if (!detections) {
                    return res.status(400).json({ error: 'No face detected' });
                }

                const query = `SELECT id_cliente, name, ci, descriptor FROM users`;
                db.all(query, [], (err, rows) => {
                    if (err) {
                        return res.status(500).json({ error: err.message });
                    }

                    let bestMatch = null;
                    let bestDistance = Infinity;

                    rows.forEach(row => {
                        const dbDescriptor = new Float32Array(JSON.parse(row.descriptor));
                        const distance = faceapi.euclideanDistance(detections.descriptor, dbDescriptor);
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestMatch = `${row.id_cliente} - ${row.name} - ${row.ci}`;
                        }
                    });

                    if (bestDistance < 0.6) {
                        res.json({ match: bestMatch });
                    } else {
                        res.json({ match: null });
                    }
                });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        };

        img.onerror = (err) => {
            res.status(400).json({ error: 'Failed to load image' });
        };
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, async () => {
    await loadModels();
    console.log(`Servidor corriendo en http://localhost:${port}`);
});
