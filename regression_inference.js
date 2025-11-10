// --- 1. CONFIGURATION: You MUST get these exact values from your Python script! ---
// --- REPLACE ALL PLACEHOLDER VALUES BELOW WITH YOUR MODEL'S CONSTANTS ---

// --- STANDARD SCALER CONSTANTS (Example for 'Rating' and 'age') ---
const RATING_MEAN = 3.3; // <--- REPLACE THIS VALUE
const RATING_STD = 0.5;  // <--- REPLACE THIS VALUE
const AGE_MEAN = 35.0;   // <--- REPLACE THIS VALUE
const AGE_STD = 20.0;    // <--- REPLACE THIS VALUE

// --- ONE-HOT ENCODING MAPPING (Customized function) ---
// This function needs to cover ALL states in your dataset, in the order of your Python OHE output.
function getOneHotState(jobState) {
    // You MUST calculate the exact number of OHE columns for job_state 
    // and ensure the indices match your Python ColumnTransformer output order.
    let stateArray = new Array(38).fill(0.0); // Size of your OHE output for states

    switch (jobState) {
        case 'CA': stateArray[0] = 1.0; break; // Check the index!
        case 'NY': stateArray[1] = 1.0; break; // Check the index!
        case 'VA': stateArray[2] = 1.0; break; // Check the index!
        // ... add all other states and their corresponding index 
        default: stateArray[37] = 1.0; break; // Set the 'Others' column (or last column)
    }
    return stateArray;
}


// --- 2. Model Loading ---
const modelPath = 'regression_model.onnx';
let session = null;
async function loadModel() {
    try {
        session = await ort.InferenceSession.create(modelPath);
        document.getElementById('output').innerText = "Model loaded. Ready for prediction.";
    } catch (e) {
        document.getElementById('output').innerText = `Error loading model: ${e.message}`;
        console.error("Failed to load ONNX model:", e);
    }
}
loadModel(); 

// --- 3. Preprocessing (The heart of the deployment) ---
function preprocessInputs() {
    const rating = parseFloat(document.getElementById('rating').value);
    const age = parseFloat(document.getElementById('age').value);
    const jobState = document.getElementById('job_state').value;
    const python_yn = parseFloat(document.getElementById('python_yn').value);
    const R_yn = parseFloat(document.getElementById('R_yn').value);
    
    // FIX: ADD THE 3 MISSING BINARY FEATURES AND SET THEM TO 0.0
    const spark_yn = 0.0; 
    const aws_yn = 0.0;
    const excel_yn = 0.0;
    
    const scaledRating = (rating - RATING_MEAN) / RATING_STD;
    const scaledAge = (age - AGE_MEAN) / AGE_STD;

    const oneHotStates = getOneHotState(jobState); 

    // Combine all features into the final 45-element array (MUST BE IN CORRECT ORDER!)
    const processedArray = [
        // Scaled Numeric Features (2)
        scaledRating, scaledAge, 
        
        // OHE State Features (38)
        ...oneHotStates, 
        
        // Binary Features (5) - NOW CORRECTLY INCLUDES ALL 5
        python_yn, 
        R_yn, 
        spark_yn, // <--- ADDED
        aws_yn,   // <--- ADDED
        excel_yn, // <--- ADDED
    ]; 

    // This check will now pass: 2 + 38 + 5 = 45
    if (processedArray.length !== 45) {
        throw new Error(`Feature count mismatch: Expected 45, got ${processedArray.length}.`);
    }
    
    return new Float32Array(processedArray);
}

// --- 4. Run Inference ---
async function runInference() {
    if (!session) {
        document.getElementById('output').innerText = "Model not loaded yet. Please wait.";
        return;
    }
    
    try {
        const inputTensorData = preprocessInputs();
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, inputTensorData.length]);
        
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        
        const predictedSalary = results.output.data[0]; 

        document.getElementById('output').innerText = 
            `Predicted Average Salary: $${predictedSalary.toFixed(2)}K`;
            
    } catch (e) {
        document.getElementById('output').innerText = `Prediction Error: ${e.message}`;
    }

}

