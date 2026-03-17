// 画像が選択された時にプレビューを表示する関数
function previewImage(input) {
    const previewArea = document.getElementById('preview-area');
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewArea.innerHTML = `
                <p>選択された画像:</p>
                <img src="${e.target.result}" style="max-width: 200px; border: 2px solid #ddd; border-radius: 5px;">
            `;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// 送信ボタンが押された時にローディングを出す関数
function showLoading() {
    const loadingDiv = document.getElementById("loading");
    if (loadingDiv) {
        loadingDiv.style.display = "block";
    }
}