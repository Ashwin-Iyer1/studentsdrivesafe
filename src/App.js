import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [predictionText, setPredictionText] = useState('');
  const [sensorData, setSensorData] = useState({
    accelerationX: null,
    accelerationY: null,
    accelerationZ: null,
    gyroX: null,
    gyroY: null,
    gyroZ: null,
  });

useEffect(() => {
  let intervalId;

  // Function to handle device motion data
  const handleDeviceMotion = (event) => {
    const { acceleration, rotationRate } = event;
    clearInterval(intervalId);
    // Update sensor data with a delay of 500 milliseconds
    intervalId = setInterval(() => {
      setSensorData({
        accelerationX: acceleration.x,
        accelerationY: acceleration.y,
        accelerationZ: acceleration.z,
        gyroX: rotationRate.alpha,
        gyroY: rotationRate.beta,
        gyroZ: rotationRate.gamma,
      });
    }, 50);
  };
    // Add event listeners for device motion
    window.addEventListener('devicemotion', handleDeviceMotion);

    // Clean up the event listeners when the component is unmounted
    return () => {
      window.removeEventListener('devicemotion', handleDeviceMotion);
      clearInterval(intervalId);
    };
  }, []); // Empty dependency array means this effect runs once after the initial render

  
  const handleButtonClick = async () => {
    const model = await tf.loadLayersModel('New_Model.h5');
    model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 32, returnSequences: true }), mergeMode: 'concat', inputShape: [100, 6] }));
    model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 32, returnSequences: true }), mergeMode: 'concat' }));
    model.add(tf.layers.lstm({ units: 32, returnSequences: false }));
    model.add(tf.layers.dropout({ rate: 0.4 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    const new_data = tf.tensor2d([[-3.6770275, 1.3274183, 6.142731, 0.13962343, 0.092566445, -0.16208291]]);
    const new_data_sequence = new_data.tile([100, 1]);
    const reshapedData = new_data_sequence.reshape([1, 100, 6]);

    const predictions = model.predict(reshapedData);
    const predictionValues = await predictions.array(); // Convert predictions to a JavaScript array

    setPredictionText(`Predictions: ${JSON.stringify(predictionValues)}`);
  };
    return (
    <div className="App">
      <p>{predictionText}</p>
      <p>Acceleration-X: {sensorData.accelerationX}</p>
      <p>Acceleration-Y: {sensorData.accelerationY}</p>
      <p>Acceleration-Z: {sensorData.accelerationZ}</p>
      <p>Gyro-X: {sensorData.gyroX}</p>
      <p>Gyro-Y: {sensorData.gyroY}</p>
      <p>Gyro-Z: {sensorData.gyroZ}</p>
      <button onClick={handleButtonClick}>Print Predictions</button>
    </div>
  );
}

export default App;
