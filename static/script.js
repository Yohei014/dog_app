document.addEventListener("DOMContentLoaded", () => {

const dropArea = document.getElementById("drop-area")
const fileInput = document.getElementById("file-input")
const previewArea = document.getElementById("preview-area")

if(dropArea){

dropArea.addEventListener("click",()=>fileInput.click())

dropArea.addEventListener("dragover",e=>{
e.preventDefault()
dropArea.classList.add("drag")
})

dropArea.addEventListener("dragleave",()=>{
dropArea.classList.remove("drag")
})

dropArea.addEventListener("drop",e=>{

e.preventDefault()

dropArea.classList.remove("drag")

const file=e.dataTransfer.files[0]

fileInput.files=e.dataTransfer.files

showPreview(file)

})

}

if(fileInput){

fileInput.addEventListener("change",()=>{

const file=fileInput.files[0]

showPreview(file)

})

}

function showPreview(file){

const reader=new FileReader()

reader.onload=function(e){

previewArea.innerHTML=`

<p>選択された画像</p>

<img src="${e.target.result}" class="preview-img">

<p>${file.name}</p>

`

}

reader.readAsDataURL(file)

}

const slider=document.getElementById("cam-slider")

if(slider){

slider.addEventListener("input",function(){

const heatmap=document.getElementById("gradcam")

heatmap.style.opacity=this.value/100

})

}

})

function showLoading(){

document.getElementById("loading").style.display="block"

}