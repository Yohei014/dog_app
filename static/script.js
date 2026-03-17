const slider = document.getElementById("camSlider")

if(slider){

slider.addEventListener("input", function(){

const heatmap = document.getElementById("heatmap")

heatmap.style.opacity = this.value / 100

})

}


const dropArea = document.getElementById("drop-area")
const fileInput = document.getElementById("fileInput")

if(dropArea){

dropArea.addEventListener("click", () => fileInput.click())

dropArea.addEventListener("dragover", e => {

e.preventDefault()
dropArea.style.borderColor = "#007aff"

})

dropArea.addEventListener("dragleave", () => {

dropArea.style.borderColor = "#ccc"

})

dropArea.addEventListener("drop", e => {

e.preventDefault()

fileInput.files = e.dataTransfer.files

})

}