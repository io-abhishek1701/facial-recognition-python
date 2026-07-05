/* ==========================================================
   QuickFace AI
   Premium Dashboard Script
   Part 1
========================================================== */

const API = window.API_BASE_URL || "http://127.0.0.1:8000";

/* ==========================================================
   DOM Elements
========================================================== */

const loadingOverlay =
    document.getElementById("loadingOverlay");

const enrollMessage =
    document.getElementById("enrollMessage");

const recognitionResult =
    document.getElementById("recognitionResult");

const deleteMessage =
    document.getElementById("deleteMessage");

const totalPersons =
    document.getElementById("totalPersons");

/* ==========================================================
   Loading
========================================================== */

function showLoading() {

    if (loadingOverlay) {

        loadingOverlay.classList.add("active");

    }

}

function hideLoading() {

    if (loadingOverlay) {

        loadingOverlay.classList.remove("active");

    }

}

/* ==========================================================
   Toast Message
========================================================== */

function showMessage(id, message, success = true) {

    const box = document.getElementById(id);

    if (!box) {

        return;

    }

    box.style.display = "block";

    box.innerHTML = message;

    if (success) {

        box.className = "message success";

    }

    else {

        box.className = "message error";

    }

    setTimeout(() => {

        box.style.display = "none";

    }, 5000);

}

/* ==========================================================
   API Wrapper
========================================================== */

async function api(url, method = "GET", body = null) {

    showLoading();

    try {

        const response = await fetch(

            API + url,

            {

                method,

                body,

                headers: {

                    Accept: "application/json"

                }

            }

        );

        const text = await response.text();

        const data = text ? JSON.parse(text) : {};

        hideLoading();

        if (!response.ok) {

            return {

                success: false,

                message: data.message || data.detail || `Request failed (${response.status})`

            };

        }

        return data;

    }

    catch (error) {

        hideLoading();

        console.error(error);

        alert(
            "Unable to connect to backend. Make sure FastAPI is running on " + API
        );

        return null;

    }

}

/* ==========================================================
   Image Preview
========================================================== */

function previewImage(inputId, previewId) {

    const input = document.getElementById(inputId);

    const preview = document.getElementById(previewId);

    input.addEventListener("change", () => {

        const file = input.files[0];

        if (!file) {

            preview.style.display = "none";

            return;

        }

        const reader = new FileReader();

        reader.onload = function (e) {

            preview.src = e.target.result;

            preview.style.display = "block";

        }

        reader.readAsDataURL(file);

    });

}

previewImage(

    "enrollImage",

    "enrollPreview"

);

previewImage(

    "recognizeImage",

    "recognizePreview"

);

/* ==========================================================
   Enroll Person
   Part 2
========================================================== */

async function enrollPerson() {

    const name = document.getElementById("name").value.trim();

    const images = document.getElementById("enrollImage").files;

    if (name === "") {

        showMessage(
            "enrollMessage",
            "Please enter the person's name.",
            false
        );

        return;
    }

    if (images.length === 0) {

        showMessage(
            "enrollMessage",
            "Please select at least one image.",
            false
        );

        return;

    }

    if (images.length > 5) {

        showMessage(
            "enrollMessage",
            "Maximum 5 images allowed.",
            false
        );

        return;

    }

    const formData = new FormData();

    formData.append("name", name);

    for (const image of images) {

        formData.append("images", image);

    }

    const data = await api(
        "/enroll",
        "POST",
        formData
    );

    if (!data)
        return;

    if (data.success) {

        showMessage(
            "enrollMessage",
            `
    <strong>✅ Enrollment Complete</strong>

    <br><br>

    <strong>Name:</strong> ${data.name}

    <br>

    <strong>Person ID:</strong> ${data.id}

    <br>

    <strong>Images Processed:</strong> ${data.images_processed}

    <br><br>

    Face embeddings generated successfully.
    `,
            true
        );

        document.getElementById("name").value = "";

        document.getElementById("enrollImage").value = "";

        document.getElementById("enrollPreview").style.display = "none";

        loadPersons();

    }

    else {

        showMessage(

            "enrollMessage",

            data.message ||

            "Enrollment Failed",

            false

        );

    }

}

/* ==========================================================
   Reset Enroll Form
========================================================== */

function resetEnrollForm() {

    document.getElementById("name").value = "";

    document.getElementById("enrollImage").value = "";

    document.getElementById("enrollPreview").style.display = "none";

}

/* ==========================================================
   Auto Refresh Dashboard
========================================================== */

async function refreshDashboard() {

    await loadPersons();

}

/* ==========================================================
   Initialize Dashboard
========================================================== */

window.addEventListener(

    "load",

    () => {

        refreshDashboard();

    }

);

/* ==========================================================
   Face Recognition
   Part 3
========================================================== */

async function recognizePerson() {

    const image =
        document.getElementById("recognizeImage").files[0];

    if (!image) {

        showMessage(
            "recognitionResult",
            "Please select an image.",
            false
        );

        return;
    }

    const formData = new FormData();

    formData.append(
        "image",
        image
    );

    const data = await api(
        "/recognize",
        "POST",
        formData
    );

    if (!data)
        return;

    if (data.success) {

        recognitionResult.innerHTML = `

        <div class="badge badge-primary">

            MATCH FOUND

        </div>

        <br><br>

        <h3>

            👤 ${data.name}

        </h3>

        <br>

        <p>

            Confidence

        </p>

        <h2>

            ${data.confidence}%

        </h2>

        <br>

        <p>

            Status :
            <strong class="text-success">

                Known Person

            </strong>

        </p>

        `;

        recognitionResult.className =
            "message success";

        recognitionResult.style.display = "block";

    }

    else {

        recognitionResult.innerHTML = `

        <div class="badge badge-danger">

            NO MATCH

        </div>

        <br><br>

        <h3>

            ❌ Unknown Person

        </h3>

        <p>

            ${data.message || "No matching person found"}

        </p>

        <br>

        <br>

        <p>

            Highest Similarity

        </p>

        <h2>

            ${data.confidence}%

        </h2>

        <br>

        <p>

            Status :
            <strong class="text-danger">

                Unknown

            </strong>

        </p>

        `;

        recognitionResult.className =
            "message error";

        recognitionResult.style.display = "block";

    }

}

/* ==========================================================
   Clear Recognition
========================================================== */

function clearRecognition() {

    recognitionResult.style.display = "none";

    document.getElementById(
        "recognizeImage"
    ).value = "";

    document.getElementById(
        "recognizePreview"
    ).style.display = "none";

}

/* ==========================================================
   Auto Hide Recognition
========================================================== */

function hideRecognitionAfterDelay() {

    setTimeout(() => {

        recognitionResult.style.display = "none";

    }, 8000);

}

/* ==========================================================
   Load All Persons
   Part 4
========================================================== */

async function loadPersons() {

    const data = await api("/persons");

    if (!data)
        return;

    const table =
        document.getElementById("personsTable");

    table.innerHTML = "";

    totalPersons.innerHTML = data.length;

    if (data.length === 0) {

        table.innerHTML = `

        <tr>

            <td colspan="4">

                No persons enrolled yet.

            </td>

        </tr>

        `;

        return;
    }

    data.forEach(person => {

        const row = document.createElement("tr");

        row.innerHTML = `

        <td>

            ${person.id}

        </td>

        <td>

            ${person.name}

        </td>

        <td>

            ${new Date(
            person.created_at
        ).toLocaleString()}

        </td>

        <td>

            <button
                class="delete-btn"
                onclick="deletePerson(${person.id})">

                Delete

            </button>

        </td>

        `;

        table.appendChild(row);

    });

}

/* ==========================================================
   Refresh List
========================================================== */

async function refreshPersons() {

    await loadPersons();

}

/* ==========================================================
   Dashboard Counter
========================================================== */

function updateCounter(count) {

    totalPersons.innerHTML = count;

}

/* ==========================================================
   Refresh Every 30 Seconds
========================================================== */

setInterval(() => {

    loadPersons();

}, 30000);

/* ==========================================================
   Search Person
========================================================== */

function filterPersons() {

    const input = document
        .getElementById("searchPerson");

    if (!input)
        return;

    const filter = input.value.toLowerCase();

    const rows = document
        .querySelectorAll("#personsTable tr");

    rows.forEach(row => {

        const text = row.innerText.toLowerCase();

        row.style.display =

            text.includes(filter)

                ? ""

                : "none";

    });

}

/* ==========================================================
   Delete Person
   Part 5
========================================================== */

async function deletePerson(id = null) {

    let personId = id;

    if (!personId) {

        personId = document
            .getElementById("deleteId")
            .value;

    }

    if (!personId) {

        showMessage(
            "deleteMessage",
            "Please enter a Person ID.",
            false
        );

        return;

    }

    const confirmDelete = confirm(
        "Are you sure you want to delete this person?"
    );

    if (!confirmDelete)
        return;

    const data = await api(
        `/person/${personId}`,
        "DELETE"
    );

    if (!data)
        return;

    if (data.success) {

        showMessage(

            "deleteMessage",

            `
            ✅ Person Deleted Successfully
            `,

            true

        );

        document
            .getElementById("deleteId")
            .value = "";

        loadPersons();

    }

    else {

        showMessage(

            "deleteMessage",

            data.message ||
            "Unable to delete person.",

            false

        );

    }

}

/* ==========================================================
   Backend Health Check
========================================================== */

async function checkBackendHealth() {

    try {

        const response = await fetch(API + "/health");

        const data = await response.json();

        document.getElementById("backendStatus").innerHTML =
            "🟢 " + data.status;

        document.getElementById("databaseStatus").innerHTML =
            "🟢 " + data.database;

        document.getElementById("modelStatus").innerHTML =
            "🟢 " + data.model;

        document.getElementById("detectorStatus").innerHTML =
            "🟢 " + data.detector;

    }

    catch (error) {

        document.getElementById("backendStatus").innerHTML =
            "🔴 Offline";

        document.getElementById("databaseStatus").innerHTML =
            "❌ Unknown";

        document.getElementById("modelStatus").innerHTML =
            "❌ Unknown";

        document.getElementById("detectorStatus").innerHTML =
            "❌ Unknown";

    }

}

/* ==========================================================
   Auto Health Check
========================================================== */

setInterval(() => {

    checkBackendHealth();

}, 10000);

/* ==========================================================
   Initial Health Check
========================================================== */

window.addEventListener(

    "load",

    () => {

        checkBackendHealth();

    }

);

/* ==========================================================
   Refresh Everything
========================================================== */

async function refreshDashboardData() {

    await checkBackendHealth();

    await loadPersons();

}

/* ==========================================================
   QuickFace AI
   Premium Dashboard Script
   Part 6
========================================================== */

/* ==========================================================
   Keyboard Shortcuts
========================================================== */

document.addEventListener("keydown", function (event) {

    // Ctrl + R -> Refresh Persons
    if (event.ctrlKey && event.key === "r") {

        event.preventDefault();

        loadPersons();

    }

    // ESC -> Hide Messages
    if (event.key === "Escape") {

        enrollMessage.style.display = "none";

        recognitionResult.style.display = "none";

        deleteMessage.style.display = "none";

    }

});


/* ==========================================================
   Copy Person ID
========================================================== */

function copyPersonId(id) {

    navigator.clipboard.writeText(id);

    showMessage(

        "deleteMessage",

        `Person ID ${id} copied to clipboard.`,

        true

    );

}


/* ==========================================================
   Reset Forms
========================================================== */

function resetAllForms() {

    // Enroll

    document.getElementById("name").value = "";

    document.getElementById("enrollImage").value = "";

    document.getElementById("enrollPreview").style.display = "none";


    // Recognition

    document.getElementById("recognizeImage").value = "";

    document.getElementById("recognizePreview").style.display = "none";


    // Delete

    const deleteInput = document.getElementById("deleteId");

    if (deleteInput) {

        deleteInput.value = "";

    }

}


/* ==========================================================
   Drag & Drop Upload
========================================================== */

function enableDragDrop(inputId) {

    const input = document.getElementById(inputId);

    if (!input) return;

    input.addEventListener("dragover", (e) => {

        e.preventDefault();

        input.style.borderColor = "#2563eb";

    });

    input.addEventListener("dragleave", () => {

        input.style.borderColor = "#d1d5db";

    });

    input.addEventListener("drop", (e) => {

        e.preventDefault();

        input.files = e.dataTransfer.files;

        input.dispatchEvent(new Event("change"));

        input.style.borderColor = "#d1d5db";

    });

}

enableDragDrop("enrollImage");

enableDragDrop("recognizeImage");


/* ==========================================================
   Scroll To Top Button
========================================================== */

const topButton = document.getElementById("topBtn");

window.addEventListener("scroll", () => {

    if (window.scrollY > 250) {

        topButton.style.display = "flex";

    }

    else {

        topButton.style.display = "none";

    }

});


/* ==========================================================
   Startup
========================================================== */

window.addEventListener("load", async () => {

    console.log("QuickFace AI Started");

    await checkBackendHealth();

    await loadPersons();

});


/* ==========================================================
   Auto Refresh Every Minute
========================================================== */

setInterval(async () => {

    await loadPersons();

}, 60000);


/* ==========================================================
   Global Error Handler
========================================================== */

window.addEventListener("error", (e) => {

    console.error(e);

});


/* ==========================================================
   Console Banner
========================================================== */

console.log(
    `
===========================================
        QuickFace AI
===========================================

Frontend Connected

Backend : FastAPI

Database : SQLite

Recognition : FaceNet

Detector : MTCNN

===========================================
`
);

/* ==========================================================
   Live Webcam Recognition
========================================================== */

let videoStream = null;

let recognitionInterval = null;

const video = document.getElementById("video");


const cameraResult = document.getElementById("cameraResult");

const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");


/* ==========================================================
   Start Camera
========================================================== */

async function startCamera() {

    try {

        videoStream = await navigator.mediaDevices.getUserMedia({

            video: {
                width: 640,
                height: 480,
                facingMode: "user"
            },

            audio: false

        });

        video.srcObject = videoStream;

        video.onloadedmetadata = () => {

            overlay.width =

                video.videoWidth;

            overlay.height =

                video.videoHeight;

            canvas.width =

                video.videoWidth;

            canvas.height =

                video.videoHeight;

        };

        cameraResult.innerHTML = "🟢 Camera Started";

        recognitionInterval = setInterval(

            recognizeLive,

            1000

        );

    }

    catch (error) {

        console.error(error);

        cameraResult.innerHTML = "❌ Unable to access camera.";

    }

}


/* ==========================================================
   Stop Camera
========================================================== */

function stopCamera() {

    if (videoStream) {

        videoStream.getTracks().forEach(track => {

            track.stop();

        });

    }

    clearInterval(recognitionInterval);

    cameraResult.innerHTML = "Camera Stopped.";

}


/* ==========================================================
   Capture Frame
========================================================== */

function captureFrame() {

    ctx.drawImage(

        video,

        0,

        0,

        canvas.width,

        canvas.height

    );

}

/* ==========================================================
   Live Recognition
========================================================== */

async function recognizeLive() {

    if (!videoStream)
        return;

    captureFrame();

    canvas.toBlob(async function (blob) {

        const formData = new FormData();

        formData.append(
            "image",
            blob,
            "frame.jpg"
        );

        try {

            const response = await fetch(
                API + "/recognize",
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            overlayCtx.clearRect(

                0,

                0,

                overlay.width,

                overlay.height

            );

            if (data.box) {

                const [x, y, w, h] = data.box;

                overlayCtx.lineWidth = 4;

                overlayCtx.strokeStyle =

                    data.success

                        ? "#00ff00"

                        : "#ff0000";

                overlayCtx.strokeRect(

                    x,

                    y,

                    w,

                    h

                );

                overlayCtx.fillStyle =

                    data.success

                        ? "#00ff00"

                        : "#ff0000";

                overlayCtx.font =

                    "20px Arial";

                overlayCtx.fillText(

                    data.success

                        ?

                        `${data.name} (${data.confidence}%)`

                        :

                        `Unknown (${data.confidence}%)`,

                    x,

                    y - 10

                );

            }

            if (data.success) {

                cameraResult.className =
                    "message success";

                cameraResult.innerHTML = `

                <h3>✅ ${data.name}</h3>

                <p>

                    Confidence :
                    <strong>${data.confidence}%</strong>

                </p>

                `;

            }

            else {

                cameraResult.className =
                    "message error";

                cameraResult.innerHTML = `

                <h3>❌ Unknown Person</h3>

                <p>

                    Confidence :
                    <strong>${data.confidence}%</strong>

                </p>

                `;

            }

        }

        catch (error) {

            console.error(error);

            cameraResult.className =
                "message error";

            cameraResult.innerHTML =

                "Backend Offline";

        }

    }, "image/jpeg", 0.8);

}