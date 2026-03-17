document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const previewArea = document.getElementById("preview-area");
    const dropText = document.getElementById("drop-text");

    // エリアクリックでファイル選択を開く
    dropArea.addEventListener("click", () => fileInput.click());

    // ドラッグ時の挙動
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files; // 入力欄にファイルをセット
            showPreview(files[0]);
        }
    });

    // ファイル選択後のプレビュー
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            showPreview(fileInput.files[0]);
        }
    });

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            dropText.style.display = "none"; // 文字を消す
            previewArea.innerHTML = `<img src="${e.target.result}">`;
        };
        reader.readAsDataURL(file);
    }
});

function showLoading() {
    document.getElementById("loading").style.display = "block";
}