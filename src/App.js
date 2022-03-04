import React, { Component } from 'react';
// import logo from './logo.svg';
import './App.css';
import axios from 'axios';

class App extends Component {

  // Constructor
  constructor() {
    super()
    this.state = {
      previewImageUrl: false,
      imageHeight: 200,
      imagePrediction: "",
    }
    this.generatePreviewImageUrl = this.generatePreviewImageUrl.bind(this)
    this.handleChange = this.handleChange.bind(this)
    this.uploadHandler = this.uploadHandler.bind(this)
  }

    // Function for previewing the chosen image
    generatePreviewImageUrl(file, callback) {
      const reader = new FileReader()
      const url = reader.readAsDataURL(file)
      reader.onloadend = e => callback(reader.result)
    }

    // Event handler when image is chosen
    handleChange(event) {
      const file = event.target.files[0]
      
      // If the image upload is cancelled
      if (!file) {
        return
      }

      this.setState({imageFile: file})
      this.generatePreviewImageUrl(file, previewImageUrl => {
            this.setState({
              previewImageUrl,
              imagePrediction:""
            })
          })
    }

    // Function for sending image to the backend
    uploadHandler(e) {
    var self = this;
    const formData = new FormData()
    formData.append('file', this.state.imageFile, 'img.png')
    
    var t0 = performance.now();
    axios.post('https://krestine.pythonanywhere.com/upload', formData)
    .then(function(response, data) {
            data = response.data;
            self.setState({imagePrediction:data})
            var t1 = performance.now();
            console.log("The time it took to predict the image " + (t1 - t0) + " milliseconds.")
        })
    }

  render() {
    return (
      <div className="App">
        <header className="App-header">
        <div className="App-upload">
          <p>
            분석할 이미지를 업로드하세요
          </p>

          {/* Button for choosing an image */}
          <div>
          <input type="file" name="file" onChange={this.handleChange} />
          </div>

          {/* Button for sending image to backend */}
          <div>
          <input type="submit" onClick={this.uploadHandler} />
          </div>

          {/* Field for previewing the chosen image */}
          <div>
          { this.state.previewImageUrl &&
          <img height={this.state.imageHeight} alt="" src={this.state.previewImageUrl} />
          }
          </div>

          {/* Text for model prediction */}
          <div>
          { this.state.imagePrediction &&
            <p>이것은 {this.state.imagePrediction} 입니다.
            </p>

          }
          </div>
          </div>
        </header>
      </div>
    );
  }
}

export default App;
