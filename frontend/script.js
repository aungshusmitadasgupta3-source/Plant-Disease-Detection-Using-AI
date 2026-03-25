const API_URL = "http://127.0.0.1:8000/predict";

let lastResult = null;

// 🌙 THEME TOGGLE
const body = document.body;
body.classList.add("dark");

document.getElementById("themeToggle").onclick = () => {
  if (body.classList.contains("dark")) {
    body.classList.replace("dark", "light");
  } else {
    body.classList.replace("light", "dark");
  }
};

// 📊 Analytics
let stats = JSON.parse(localStorage.getItem("stats")) || {
  total: 0,
  healthy: 0,
  diseased: 0
};

function renderStats() {
  document.getElementById("total").innerText = stats.total;
  document.getElementById("healthy").innerText = stats.healthy;
  document.getElementById("diseased").innerText = stats.diseased;
}

function updateStats(isHealthy) {
  stats.total++;
  isHealthy ? stats.healthy++ : stats.diseased++;
  localStorage.setItem("stats", JSON.stringify(stats));
  renderStats();
}

renderStats();

// 🖼️ Preview
document.getElementById("fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = document.getElementById("preview");

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  }
});

// 🚀 Predict
async function predict() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Upload image");

  let formData = new FormData();
  formData.append("file", file);

  let res = await fetch(API_URL, {
    method: "POST",
    body: formData
  });

  let data = await res.json();
  if (data.error) return alert(data.error);

  lastResult = data;

  renderResult(data);
}

function renderResult(data) {
  const isHealthy = data.disease.toLowerCase().includes("healthy");
  updateStats(isHealthy);

  document.getElementById("result").classList.remove("hidden");

  document.getElementById("disease").innerText =
    `${data.crop} - ${data.disease}`;

  document.getElementById("confidence").innerText =
    `Confidence: ${data.confidence}%`;

  document.getElementById("remedy").innerText = data.remedy;

  document.getElementById("explanation").innerText =
    "AI detected patterns like discoloration and spots.";
}

// 📄 DOWNLOAD PDF
function downloadPDF() {
  if (!lastResult) return alert("No result");

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  doc.setFontSize(18);
  doc.text("Plant Disease Report", 20, 20);

  doc.setFontSize(12);
  doc.text(`Crop: ${lastResult.crop}`, 20, 40);
  doc.text(`Disease: ${lastResult.disease}`, 20, 50);
  doc.text(`Confidence: ${lastResult.confidence}%`, 20, 60);

  doc.text("Remedy:", 20, 80);
  doc.text(lastResult.remedy, 20, 90);

  doc.save("plant_report.pdf");
}