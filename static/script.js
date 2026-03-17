document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const previewArea = document.getElementById("preview-area");

    // クリックでファイル選択を開く
    dropArea.addEventListener("click", () => fileInput.click());

    // ドラッグ＆ドロップイベント
    ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
        dropArea.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add("dragover"), false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove("dragover"), false);
    });

    dropArea.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFiles(files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files[0]);
        }
    });

    function handleFiles(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewArea.innerHTML = `
                <img src="${e.target.result}" style="max-width: 100%; border-radius: 12px;">
                <p style="margin-top:10px; color:#76B55B; font-weight:bold;">${file.name}</p>
            `;
        };
        reader.readAsDataURL(file);
    }
});

function showLoading() {
    document.getElementById("loading").style.display = "block";
    document.querySelector(".btn").style.display = "none";
}