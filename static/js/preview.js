document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const dropzone = document.getElementById("dropzone");
    const placeholder = document.getElementById("previewPlaceholder");
    const fileName = document.getElementById("fileName");

    if (!fileInput || !preview) {
        return;
    }

    const renderPreview = (file) => {
        if (!file || !file.type.startsWith("image/")) {
            preview.src = "";
            preview.classList.remove("active");
            if (placeholder) {
                placeholder.style.display = "block";
            }
            if (fileName) {
                fileName.textContent = "No file selected";
            }
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target.result;
            preview.classList.add("active");
            if (placeholder) {
                placeholder.style.display = "none";
            }
        };
        reader.readAsDataURL(file);

        if (fileName) {
            fileName.textContent = file.name;
        }
    };

    fileInput.addEventListener("change", () => {
        renderPreview(fileInput.files[0]);
    });

    if (!dropzone) {
        return;
    }

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("drag-active");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const files = event.dataTransfer.files;
        if (!files || !files.length) {
            return;
        }

        const transfer = new DataTransfer();
        transfer.items.add(files[0]);
        fileInput.files = transfer.files;
        renderPreview(files[0]);
    });
});
