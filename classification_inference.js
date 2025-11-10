// --- 1. CONFIGURATION: You MUST get these exact values from your Python script! ---
// --- REPLACE ALL PLACEHOLDER VALUES BELOW WITH YOUR MODEL'S CONSTANTS ---

// --- STANDARD SCALER CONSTANTS (Example for 'avg_salary' and 'Rating') ---
const AVG_SALARY_MEAN = 100.0; // <--- REPLACE THIS VALUE
const AVG_SALARY_STD = 30.0;   // <--- REPLACE THIS VALUE
const RATING_MEAN_CLS = 3.3;   // <--- REPLACE THIS VALUE
const RATING_STD_CLS = 0.5;    // <--- REPLACE THIS VALUE
const MIN_SALARY_MEAN = 75.0;  // <--- REPLACE THIS VALUE (min_salary)
const MIN_SALARY_STD = 25.0;   // <--- REPLACE THIS VALUE (min_salary)


// --- OHE MAPPING FUNCTION (Conceptual - Must be fully customized) ---
// This function must generate the 47 OHE features from 'job_state' and 'Size' 
// in the EXACT order your Python ColumnTransformer created them.
function getOneHotMapping(jobState, companySize) {
    // Total OHE features for classification is 47.
    let OHE_features = new Array(47).fill(0.0);
    
    // --- Custom Logic for setting the 1.0 at the correct index for jobState and companySize ---
    // Example (You must fill in the correct indices for ALL states and size categories):
    
    // States (Assume states occupy indices 0 through 30)
    switch (jobState) {
        case 'CA': OHE_features[0] = 1.0; break; 
        case 'NY': OHE_features[1] = 1.0; break; 
        // ... and so on for all states ...
    }
    
    // Size (Assume size categories occupy indices 31 through 46)
    switch (companySize) {
        case '1-50': OHE_features[31] = 1.0; break; 
        case '51-200': OHE_features[32] = 1.0; break; 
        // ... and so on for all size categories ...
    }
    
    return OHE_features;
}


// --- 2. Model Loading ---
const modelPathCls = 'classification_model.onnx';
let sessionCls = null;
async function loadModelCls() {
    try {
        sessionCls = await ort.InferenceSession.create(modelPathCls);
        document.getElementById('output_cls').innerText = "Classification Model loaded. Ready for prediction.";
    } catch (e) {
        document.getElementById('output_cls').innerText = `Error loading model: ${e.message}`;
        console.error("Failed to load Classification ONNX model:", e);
    }
}
loadModelCls(); 

// --- 3. Preprocessing (The heart of the deployment) ---
function preprocessInputsCls() {
    const avgSalary = parseFloat(document.getElementById('avg_salary').value);
    const rating = parseFloat(document.getElementById('rating_cls').value);
    const minSalary = parseFloat(document.getElementById('min_salary').value);
    // ... collect all other 2 numeric features: age, max_salary ...
    const jobState = document.getElementById('job_state_cls').value;
    const companySize = document.getElementById('size_cls').value;
    
    // Apply Standard Scaling (All 5 numeric features must be scaled)
    const scaledAvgSalary = (avgSalary - AVG_SALARY_MEAN) / AVG_SALARY_STD;
    const scaledRating = (rating - RATING_MEAN_CLS) / RATING_STD_CLS;
    const scaledMinSalary = (minSalary - MIN_SALARY_MEAN) / MIN_SALARY_STD; 
    // ... scale the other 2 numeric features: max_salary, age ...

    const oneHotFeatures = getOneHotMapping(jobState, companySize); 

    // Combine all 52 features into the final array (MUST be in correct order!)
    const processedArray = [
        // Scaled Numeric Features (5 of them)
        scaledRating, 
        /* scaled_age */
        scaledMinSalary, 
        scaledAvgSalary, 
        /* scaled_max_salary */
        
        // All OHE features (47 of them, from job_state and Size)
        ...oneHotFeatures,
    ]; 

    if (processedArray.length !== 52) {
         throw new Error(`Feature count mismatch: Expected 52, got ${processedArray.length}. Check your OHE array size.`);
    }
    
    return new Float32Array(processedArray);
}

// --- 4. Run Inference ---
async function runInferenceCls() {
    if (!sessionCls) {
        document.getElementById('output_cls').innerText = "Model not loaded yet. Please wait.";
        return;
    }
    
    try {
        const inputTensorData = preprocessInputsCls();
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, inputTensorData.length]);
        
        const feeds = { input: inputTensor };
        const results = await sessionCls.run(feeds);
        
        const probability = results.output.data[0]; 
        const prediction = probability > 0.5 ? "YES (Required)" : "NO (Not Required)";

        document.getElementById('output_cls').innerText = 
            `Python Requirement Prediction: ${prediction} (Probability: ${probability.toFixed(3)})`;
            
    } catch (e) {
        document.getElementById('output_cls').innerText = `Prediction Error: ${e.message}`;
    }
}