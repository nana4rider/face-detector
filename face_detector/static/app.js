const form = document.querySelector("form");
const result = document.getElementById("result");
let imageUrl;

for (const element of form.elements) {
  if (element.name && element.type !== "file") {
    const saved = localStorage.getItem(`face-detector:${element.name}`);
    if (saved !== null) element.value = saved;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  for (const [key, value] of formData) {
    if (typeof value === "string") {
      localStorage.setItem(`face-detector:${key}`, value);
    }
  }

  try {
    const startedAt = performance.now();
    const response = await fetch(form.action, {method: "POST", body: formData});
    const elapsed = Math.round(performance.now() - startedAt);
    if (!response.ok) {
      const error = await response.json();
      result.innerHTML = `<div class="alert alert-danger" role="alert"></div>`;
      result.firstElementChild.textContent = `エラー (${response.status}): ${error.error}`;
      return;
    }

    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(await response.blob());
    result.replaceChildren();
    const image = document.createElement("img");
    image.className = "img-fluid";
    image.src = imageUrl;
    image.alt = "Detected Face";
    const dimensions = document.createElement("p");
    dimensions.textContent = `Width: ${response.headers.get("X-Image-Width")}px, Height: ${response.headers.get("X-Image-Height")}px`;
    const timing = document.createElement("p");
    timing.textContent = `Elapsed Time: ${elapsed} ms`;
    result.append(image, dimensions, timing);
  } catch (error) {
    result.innerHTML = `<div class="alert alert-danger" role="alert"></div>`;
    result.firstElementChild.textContent = `エラー: ${error.message}`;
  }
});
