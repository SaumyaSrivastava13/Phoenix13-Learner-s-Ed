const videoElement = document.getElementById('videoElement');
const pred = document.getElementById('pred');

let stream;
let k = 0;
let csi; // Interval variable

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(videoStream => {
      stream = videoStream;
      videoElement.srcObject = stream;
      csi = setInterval(captureFrame, 2000); // Capture frame every 5 seconds
    })
    .catch(error => {
      console.error('Error accessing camera:', error);
    });
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

var excercise = ['Neck', 'Back', 'Should'];

function captureFrame() {
  // Create a canvas element
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  // Draw the current frame from the video element onto the canvas
  const context = canvas.getContext('2d');
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Get the base64-encoded image data from the canvas
  const imageData = canvas.toDataURL('image/jpeg');

  quiz_data = {
    image_data: imageData,
  };

  // Send the image data as a POST request to the Django backend using AJAX
  $.ajax({
    url: '',
    type: 'POST',
    data: JSON.stringify(quiz_data),
    contentType: 'application/json',
    success: function (response) {
      console.log(response.message);
      // Handle the response from the Django backend
      const prediction = response.prediction; // Assuming the prediction is returned as a property in the response
      if (prediction == "Yes") {
        pred.innerHTML = "Drowsiness Detected";
        clearInterval(csi); // Halt the interval
        alert(excercise[getRandomInt(excercise.length)]);
        setTimeout(() => {
          pred.innerHTML = ""; // Clear the alert message after it is clicked
          csi = setInterval(captureFrame, 2000); // Resume the interval
        }, 5000); // Adjust the duration of the alert as needed
      } else {
        pred.innerHTML = "All Ok!";
      }
      console.log('Response from Django:', prediction);
    },
    error: function (xhr, status, error) {
      console.error('Error sending image to Django:', error);
    }
  });
  console.log(k);
  if (k == 1) {
    clearInterval(csi);
    stopCamera();
  }
}

function stopCamera() {
  if (stream && stream.getTracks) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
}

startCamera();

var timer = setTimeout(() => {
  k = 1;
  console.log("Finish");
  window.location.href = "/lectures/";
}, 60000);
