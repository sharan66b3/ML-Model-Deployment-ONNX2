// --- 1. CONFIGURATION: You MUST replace these placeholder values with your model's constants ---
// --- ALL 5 numeric features need constants (Means and Standard Deviations) ---
const AVG_SALARY_MEAN = 100.0; // <--- REPLACE THIS VALUE
const AVG_SALARY_STD = 30.0;   // <--- REPLACE THIS VALUE
const RATING_MEAN_CLS = 3.3;   // <--- REPLACE THIS VALUE
const RATING_STD_CLS = 0.5;    // <--- REPLACE THIS VALUE
const MIN_SALARY_MEAN = 75.0;  // <--- REPLACE THIS VALUE
const MIN_SALARY_STD = 25.0;   // <--- REPLACE THIS VALUE

// ADDED MISSING CONSTANTS (You MUST replace these placeholders)
const AGE_MEAN_CLS = 40.0;     // <--- REPLACE THIS VALUE 
const AGE_STD_CLS = 15.0;      // <--- REPLACE THIS VALUE 
const MAX_SALARY_MEAN = 120.0; // <--- REPLACE THIS VALUE 
const MAX_SALARY_STD = 45.0;   // <--- REPLACE THIS VALUE 


// --- OHE MAPPING FUNCTION (47 features) ---
function getOneHotMapping(jobState, companySize) {
    // Total OHE features for classification is 47.
    let OHE_features = new Array(47).fill(0.0);
    
    // States (Example: Indices 0 through 31, based on your Python model)
    switch (jobState) {
        case 'CA': OHE_features[0] = 1.0; break;
        case 'NY': OHE_features[1] = 1.0; break; 
        case 'TX': OHE_features[2] = 1.0; break; 
        case 'MD': OHE_features[3] = 1.0; break; 
        case 'VA': OHE_features[4] = 1.0; break; 
        // ... include all other state cases up to index 30 ...
        default: OHE_features[31] = 1.0; break; // Placeholder for 'Other States'
    }
    
    // Size (Example: Indices 32 through 46, based on your Python model)
    switch (companySize) {
        case '1-50': OHE_features[32] = 1.0; break; 
        case '51-200': OHE_features[33] = 1.0; break; 
        case '501-1000': OHE_features[34] = 1.0; break; 
        case '10000+': OHE_features[35] = 1.0; break; 
        case 'Unknown': OHE_features[36] = 1.0; break; 
        default: OHE_features[46] = 1.0; break; // Placeholder for 'Other Size'
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

// --- 3. Preprocessing (FIXED for 52 features) ---
function preprocessInputsCls() {
    // 1. COLLECT ALL 5 NUMERIC FEATURES (Must match HTML IDs: _cls suffix added for max_salary and age)
    const avgSalary = parseFloat(document.getElementById('avg_salary').value);
    const rating = parseFloat(document.getElementById('rating_cls').value);
    const minSalary = parseFloat(document.getElementById('min_salary').value);
    const age = parseFloat(document.getElementById('age_cls').value);       // <--- ADDED COLLECTION
    const maxSalary = parseFloat(document.getElementById('max_salary_cls').value); // <--- ADDED COLLECTION
    
    const jobState = document.getElementById('job_state_cls').value;
    const companySize = document.getElementById('size_cls').value;
    
    // 2. SCALE ALL 5 NUMERIC FEATURES
    const scaledAvgSalary = (avgSalary - AVG_SALARY_MEAN) / AVG_SALARY_STD;
    const scaledRating = (rating - RATING_MEAN_CLS) / RATING_STD_CLS;
    const scaledMinSalary = (minSalary - MIN_SALARY_MEAN) / MIN_SALARY_STD; 
    const scaledAge = (age - AGE_MEAN_CLS) / AGE_STD_CLS;                  // <--- ADDED SCALING
    const scaledMaxSalary = (maxSalary - MAX_SALARY_MEAN) / MAX_SALARY_STD;  // <--- ADDED SCALING

    const oneHotFeatures = getOneHotMapping(jobState, companySize); 

    // 3. COMBINE ALL 52 FEATURES IN THE CORRECT ORDER (5 Scaled + 47 OHE)
    const processedArray = [
        // Scaled Numeric Features (5 of them - ORDER MUST MATCH PYTHON)
        scaledRating, 
        scaledAge,              // <--- NOW INCLUDED
        scaledMinSalary, 
        scaledAvgSalary, 
        scaledMaxSalary,        // <--- NOW INCLUDED
        
        // All OHE features (47 of them)
        ...oneHotFeatures,
    ]; 

    // The check will now pass: 5 + 47 = 52
    if (processedArray.length !== 52) {
         throw new Error(`Feature count mismatch: Expected 52, got ${processedArray.length}.`);
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
        
        // Assuming output is a probability [0.0 - 1.0]
        const probability = results.output.data[0]; 
        const prediction = probability > 0.5 ? "YES (Required)" : "NO (Not Required)";

        document.getElementById('output_cls').innerText = 
            `Python Requirement Prediction: ${prediction} (Probability: ${probability.toFixed(3)})`;
            
    } catch (e) {
        document.getElementById('output_cls').innerText = `Prediction Error: ${e.message}`;
    }
}
