import React, { Component } from 'react';
// import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import CanvasDraw from "react-canvas-draw";
import ReactDOM from "react-dom";
ReactDOM.render(<CanvasDraw />, document.getElementById("root"));
class App extends Component {

  // Constructor
  constructor() {
    super()
    this.state = {
      previewImageUrl: false,
      imageHeight: 200,
      imagePrediction: "",
	  exportedDrawing: "",
    }
    this.generatePreviewImageUrl = this.generatePreviewImageUrl.bind(this)
    this.handleChange = this.handleChange.bind(this)
    this.uploadHandler = this.uploadHandler.bind(this)
	this.testUploadHandler = this.testUploadHandler.bind(this)
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

	dataURLtoFile = (dataurl, filename) => {
	  var arr = dataurl.split(',')
	  var mime = arr[0].match(/:(.*?);/)[1]
	  var bstr = atob(arr[1])
	  let n = bstr.length
	  var u8arr = new Uint8Array(n)
	  while (n) {
		u8arr[n - 1] = bstr.charCodeAt(n - 1)
		n -= 1 // to make eslint happy
	  }
	  return new File([u8arr], filename, { type: mime })
	};

  handleExport = () => {
    console.log("clicked");
    var base64 = this.saveableCanvas.canvasContainer.childNodes[1].toDataURL();
    
    this.setState({exportedDrawing: base64});
  };
  
    testUploadHandler(e) {
	console.log("test upload handler");
	var base64 = this.saveableCanvas.canvasContainer.childNodes[1].toDataURL();
    var self = this;
    const formData = new FormData()
	var base64file = this.dataURLtoFile(base64)
    formData.append('file', base64file, 'img.png')

    
    var t0 = performance.now();
	axios.post('https://krestine.pythonanywhere.com/upload', formData)
    //axios.post('http://127.0.0.1:5000/upload', formData)
    .then(function(response, data) {
            data = response.data;
            self.setState({imagePrediction:data})
            var t1 = performance.now();
            console.log("The time it took to predict the image " + (t1 - t0) + " milliseconds.")
        })
    }
	
	eraseCanvas(){
		this.saveableCanvas.eraseAll();
		this.setState({ imagePrediction: null });
	};
	
    // Function for sending image to the backend
    uploadHandler(e) {
    var self = this;
    const formData = new FormData()
    formData.append('file', this.state.imageFile, 'img.png')
    
    var t0 = performance.now();
    axios.post('https://krestine.pythonanywhere.com/upload', formData)
	//axios.post('http://127.0.0.1:5000/upload', formData)
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
	          <header>
			  <div>현재 학습된 모델 :<br/>
			  사과, 산, 달, 얼굴, 문,<br/>
			  봉투, 물고기, 기타, 별, 번개</div>
		<div align="center"><CanvasDraw
          ref={canvasDraw => (this.saveableCanvas = canvasDraw)}
          brushColor={this.state.color}
          brushRadius={this.state.brushRadius}
          lazyRadius={this.state.lazyRadius}
          canvasWidth={this.state.width}
          canvasHeight={this.state.height}
        /></div>
		        <div>
          { this.state.imagePrediction &&
            <p><h2>이것은 {this.state.imagePrediction} 입니다.</h2>
            </p>

          }
          </div>
		  <br/>
	<button
            onClick={() => {
              this.eraseCanvas();
            }}
          >
            <h1>지우기</h1>
          </button>
		<button onClick={this.testUploadHandler}>
            <h1>　확인　</h1>
        </button>
        </header>
      </div>
    );
  }
}

export default App;
