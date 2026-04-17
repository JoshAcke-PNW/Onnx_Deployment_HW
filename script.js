async function runInference() {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.innerText = "Calculating...";

    try {
        const inputs = [
            parseFloat(document.getElementById('cylinders').value),
            parseFloat(document.getElementById('displacement').value),
            parseFloat(document.getElementById('horsepower').value),
            parseFloat(document.getElementById('weight').value),
            parseFloat(document.getElementById('acceleration').value),
            parseFloat(document.getElementById('year').value),
            parseFloat(document.getElementById('origin').value)
        ];

        const session = await ort.InferenceSession.create('./model.onnx');
        const inputTensor = new ort.Tensor('float32', Float32Array.from(inputs), [1, 7]);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        const output = results.output.data[0];

        resultDiv.innerText = `Predicted: ${output.toFixed(2)} MPG`;

    } catch (e) {
        console.error(e);
        resultDiv.innerText = "Error loading model. Check console.";
    }
}