document
  .getElementById("prediction-form")
  .addEventListener("submit", function (e) {
    e.preventDefault();

    let formData = new FormData(this);
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.prediction) {
          const resultDiv = document.getElementById("result");
          document.getElementById("prediction-result").innerText =
            data.prediction;
          resultDiv.classList.remove("result-hidden");
          resultDiv.style.opacity = "0";
          setTimeout(() => {
            resultDiv.style.opacity = "1";
          }, 100);
        }
      })
      .catch((error) => console.log("Error:", error));
  });
